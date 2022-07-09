import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx

# The dataset pickle and index files are in ./data/molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']

def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    g.ndata['eigvec'] = g.ndata['pos_enc']

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


def make_full_graph(g, adaptive_weighting=None):

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    # full_g.ndata['feat'] = g.ndata['feat']
    full_g.ndata['attr'] = g.ndata['attr']
    
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass
    
    try:
        full_g.ndata['eigvec'] = g.ndata['eigvec']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    if 'feat' in g.edata:
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 

    
    # This code section only apply for GraphiT --------------------------------------------
    if adaptive_weighting is not None:
        p_steps, gamma = adaptive_weighting
    
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        
        # Adaptive weighting k_ij for each edge
        if p_steps == "qtr_num_nodes":
            p_steps = int(0.25*n)
        elif p_steps == "half_num_nodes":
            p_steps = int(0.5*n)
        elif p_steps == "num_nodes":
            p_steps = int(n)
        elif p_steps == "twice_num_nodes":
            p_steps = int(2*n)

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = torch.from_numpy(k_RW_power.toarray())

        # Assigning edge features k_RW_eij for adaptive weighting during attention
        full_edge_u, full_edge_v = full_g.edges()
        num_edges = full_g.number_of_edges()

        k_RW_e_ij = []
        for edge in range(num_edges):
            k_RW_e_ij.append(k_RW_power[full_edge_u[edge], full_edge_v[edge]])

        full_g.edata['k_RW'] = torch.stack(k_RW_e_ij,dim=-1).unsqueeze(-1).float()
    # --------------------------------------------------------------------------------------
        
    return full_g



class TUDataset(torch.utils.data.Dataset):

    def __init__(self, name, fold_idx=1):
        """
            Loading TU dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        data_path = 'data/TUDataset'
        # dataset = datasets.TUDataset(data_path, name)
        # dataset = dgl.data.TUDataset(name, raw_dir=data_path)
        dataset = dgl.data.GINDataset(name, False, raw_dir=data_path)

        idx_path = 'data/fold-idx/{}/inner_folds/{}-{}-{}.txt'
        test_idx_path = 'data/fold-idx/{}/test_idx-{}.txt'

        inner_idx = 1
        train_fold_idx = torch.from_numpy(np.loadtxt(
            idx_path.format(name, 'train_idx', fold_idx, inner_idx)).astype(int))
        val_fold_idx = torch.from_numpy(np.loadtxt(
            idx_path.format(name, 'val_idx', fold_idx, inner_idx)).astype(int))
        test_fold_idx = torch.from_numpy(np.loadtxt(
            test_idx_path.format(name, fold_idx)).astype(int))

        # split_idx = dataset.get_idx_split()

        # split_idx["train"] = split_idx["train"]
        # split_idx["valid"] = split_idx["valid"]
        # split_idx["test"] = split_idx["test"]

        self.train = [dataset[idx] for idx in train_fold_idx]
        self.val = [dataset[idx] for idx in val_fold_idx]
        self.test = [dataset[idx] for idx in test_fold_idx]

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        labels = torch.stack(labels)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels, snorm_n
    

    def _add_lap_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train = [(lap_positional_encoding(g, pos_enc_dim), label) for g in self.train]
        self.val = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]
    
    def _add_eig_vecs(self, pos_enc_dim):

        # This is used if we visualize the eigvecs
        self.train = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.train]
        self.val = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.test]
    
    def _init_positional_encodings(self, pos_enc_dim, type_init):
        
        # Initializing positional encoding randomly with l2-norm 1
        self.train = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.train]
        self.val = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.val]
        self.test = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.test]
        
    def _make_full_graph(self, adaptive_weighting=None):
        self.train = [(make_full_graph(g, adaptive_weighting), label) for g, label in self.train]
        self.val = [(make_full_graph(g, adaptive_weighting), label) for g, label in self.val]
        self.test = [(make_full_graph(g, adaptive_weighting), label) for g, label in self.test]



    # def _add_lap_positional_encodings(self, pos_enc_dim):
        
    #     # Graph positional encoding v/ Laplacian eigenvectors
    #     self.train.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
    #     self.val.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
    #     self.test.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]
    
    # def _add_eig_vecs(self, pos_enc_dim):

    #     # This is used if we visualize the eigvecs
    #     self.train.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.train.graph_lists]
    #     self.val.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.val.graph_lists]
    #     self.test.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.test.graph_lists]
    
    # def _init_positional_encodings(self, pos_enc_dim, type_init):
        
    #     # Initializing positional encoding randomly with l2-norm 1
    #     self.train.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.train.graph_lists]
    #     self.val.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.val.graph_lists]
    #     self.test.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.test.graph_lists]
        
    # def _make_full_graph(self, adaptive_weighting=None):
    #     self.train.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.train.graph_lists]
    #     self.val.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.val.graph_lists]
    #     self.test.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.test.graph_lists]




