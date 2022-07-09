
import time
import os
import pickle
import numpy as np

import dgl
import torch
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx

import hashlib

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
    full_g.ndata['feat'] = g.ndata['feat']
    # full_g.ndata['attr'] = g.ndata['attr']
    
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
        # full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
        # TODO: make the edge features 1d in data loading part
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat'][:,0].long()
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

class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split,
                 dataset=None):

        self.split = split
        self.is_test = split.lower() in ['test', 'val'] 
        if dataset is None:
            with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = dataset
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()
    

    def _prepare(self):

        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        # count = 0
        for data in self.dataset:
            # count += 1
            # if count>100:
            #     break
            
            # # node_features = data.node_feat
            # node_features =  data[0].ndata['feat']
            # edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # # Create the DGL Graph
            # g = dgl.DGLGraph()
            # g.add_nodes(node_features.size(0))
            # g.ndata['feat'] = node_features.long()
            # for src, dst in edge_list:
            #     g.add_edges(src.item(), dst.item())

            # # adding edge features for Residual Gated ConvNet
            # #edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            # edge_feat_dim = 1 # dim same as node feature dim
            # g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            g = data[0]
            # g.ndata['feat'] =  g.ndata['feat'].reshape((-1,1))
            g.ndata['feat'] =  g.ndata['feat']
            self.graph_lists.append(g)
            # self.node_labels.append(data.node_label)
            self.node_labels.append(data[1])

            # g = data[0]
            # self.graph_lists.append(g)
            # # self.node_labels.append(data.node_label)
            # self.node_labels.append(data[1])


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = 'data/SBMs'
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    
class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            # self.train = f[0]
            # self.val = f[1]
            # self.test = f[2]

            self.train = load_SBMsDataSetDGL(data_dir, 'SBM_'+name, split='train', dataset=f[0])
            self.test = load_SBMsDataSetDGL(data_dir, name, split='test', dataset=f[1])
            self.val = load_SBMsDataSetDGL(data_dir, name, split='val', dataset=f[2])
                    
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        # labels = torch.stack(labels)
        labels = torch.cat(labels).long()
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

    # def collate(self, samples):

    #     graphs, labels = map(list, zip(*samples))
    #     labels = torch.cat(labels).long()
    #     batched_graph = dgl.batch(graphs)
        
    #     return batched_graph, labels
    

    # def _laplace_decomp(self, max_freqs):
    #     self.train.graph_lists = [laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
    #     self.val.graph_lists = [laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
    #     self.test.graph_lists = [laplace_decomp(g, max_freqs) for g in self.test.graph_lists]
    

    # def _make_full_graph(self):
    #     self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
    #     self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
    #     self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]


    # def _add_edge_laplace_feats(self):
    #     self.train.graph_lists = [add_edge_laplace_feats(g) for g in self.train.graph_lists]
    #     self.val.graph_lists = [add_edge_laplace_feats(g) for g in self.val.graph_lists]
    #     self.test.graph_lists = [add_edge_laplace_feats(g) for g in self.test.graph_lists]  


# class load_SBMsDataSetDGL(torch.utils.data.Dataset):

#     def __init__(self,
#                  data_dir,
#                  name,
#                  split):

#         self.split = split
#         self.is_test = split.lower() in ['test', 'val'] 
#         with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
#             self.dataset = pickle.load(f)
#         self.node_labels = []
#         self.graph_lists = []
#         self.n_samples = len(self.dataset)
#         self._prepare()
    

#     def _prepare(self):

#         print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

#         for data in self.dataset:

#             node_features = data.node_feat
#             edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

#             # Create the DGL Graph
#             g = dgl.DGLGraph()
#             g.add_nodes(node_features.size(0))
#             g.ndata['feat'] = node_features.long()
#             for src, dst in edge_list:
#                 g.add_edges(src.item(), dst.item())

#             # adding edge features for Residual Gated ConvNet
#             #edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
#             edge_feat_dim = 1 # dim same as node feature dim
#             g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

#             self.graph_lists.append(g)
#             self.node_labels.append(data.node_label)


#     def __len__(self):
#         """Return the number of graphs in the dataset."""
#         return self.n_samples

#     def __getitem__(self, idx):
#         """
#             Get the idx^th sample.
#             Parameters
#             ---------
#             idx : int
#                 The sample index.
#             Returns
#             -------
#             (dgl.DGLGraph, int)
#                 DGLGraph with node feature stored in `feat` field
#                 And its label.
#         """
#         return self.graph_lists[idx], self.node_labels[idx]


# class SBMsDatasetDGL(torch.utils.data.Dataset):

#     def __init__(self, name):
#         """
#             TODO
#         """
#         start = time.time()
#         print("[I] Loading data ...")
#         self.name = name
#         data_dir = 'data/SBMs'
#         self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
#         self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
#         self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
#         print("[I] Finished loading.")
#         print("[I] Data load time: {:.4f}s".format(time.time()-start))



# def laplace_decomp(g, max_freqs):


#     # Laplacian
#     n = g.number_of_nodes()
#     A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
#     N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
#     L = sp.eye(g.number_of_nodes()) - N * A * N

#     # Eigenvectors with numpy
#     EigVals, EigVecs = np.linalg.eigh(L.toarray())
#     EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

#     # Normalize and pad EigenVectors
#     EigVecs = torch.from_numpy(EigVecs).float()
#     EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
#     if n<max_freqs:
#         g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
#     else:
#         g.ndata['EigVecs']= EigVecs
        
    
#     #Save eigenvales and pad
#     EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
#     if n<max_freqs:
#         EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
#     else:
#         EigVals=EigVals.unsqueeze(0)
        
    
#     #Save EigVals node features
#     g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
#     return g



# def make_full_graph(g):

    
#     full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

#     #Here we copy over the node feature data and laplace encodings
#     full_g.ndata['feat'] = g.ndata['feat']

#     try:
#         full_g.ndata['EigVecs'] = g.ndata['EigVecs']
#         full_g.ndata['EigVals'] = g.ndata['EigVals']
#     except:
#         pass
    
#     #Populate edge features w/ 0s
#     full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
#     full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
#     #Copy real edge data over
#     full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
#     full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
#     return full_g


# def add_edge_laplace_feats(g):

    
#     EigVals = g.ndata['EigVals'][0].flatten()
    
#     source, dest = g.find_edges(g.edges(form='eid'))
    
#     #Compute diffusion distances and Green function
#     g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs']-g.nodes[dest].data['EigVecs']).unsqueeze(2)
#     g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
#     g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(),1).unsqueeze(2)
    
#     #No longer need EigVecs and EigVals stored as node features
#     del g.ndata['EigVecs']
#     del g.ndata['EigVals']
    
#     return g


