import torch
import scipy as sp
import numpy as np
import time
import pickle
import dgl
import sys
# sys.path.append('../LPE/')
# sys.path.append('../LPE/data/')
# from data import data

nsim=250;

def laplace_decomp(g, max_freqs):


    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g

def get_laplacian_sp(g, norm='sym'):
    n = g.number_of_nodes()
    # TODO: replace with DGLGraph.adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    return L

def make_full_graph(g):

    
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    return full_g


def add_edge_laplace_feats(g):

    
    EigVals = g.ndata['EigVals'][0].flatten()
    
    source, dest = g.find_edges(g.edges(form='eid'))
    
    #Compute diffusion distances and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs']-g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(),1).unsqueeze(2)
    
    
    #No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']
    
    return g

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading ZINC dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
    

    def _laplace_decomp(self, max_freqs):
        self.train.graph_lists = [laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
        self.val.graph_lists = [laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
        self.test.graph_lists = [laplace_decomp(g, max_freqs) for g in self.test.graph_lists]
    

    def _make_full_graph(self):
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]


    def _add_edge_laplace_feats(self):
        self.train.graph_lists = [add_edge_laplace_feats(g) for g in self.train.graph_lists]
        self.val.graph_lists = [add_edge_laplace_feats(g) for g in self.val.graph_lists]
        self.test.graph_lists = [add_edge_laplace_feats(g) for g in self.test.graph_lists]        


# load ZINC
dataset = MoleculeDataset('ZINC')
dataset._laplace_decomp(1e8)
all_graph_list = dataset.train.graph_lists + dataset.test.graph_lists + dataset.val.graph_lists
for g in all_graph_list:
    print(g)

# A=double(A);
# n=size(A,1);
# nf=size(F,2);
# d = sum(A,2);
# % normalized laplacien
# dis=1./sqrt(d);
# dis(isinf(dis))=0;
# dis(isnan(dis))=0;
# D=diag(dis);
# # nL=eye(n)-(A*D).T*D;
# laplacian_mat = get_laplacian_sp(g)
# laplacian_mat = 2.*laplacian_mat/lambda_max - sp.eye(g.number_of_nodes())
# laplacian_mat = laplacian_mat.astype(np.float32)

# [u v]=eig(nL);
# % make eignevalue as vector
# v=diag(v);

# %fprintf('u = %d ',size(u));
# A0=A+eye(n);
# bb=zeros(n,nsim);
# for i=1:nsim
#     i
#     Wq=randn(nf,8);
#     Wk=randn(nf,8);
#     Wv=randn(nf,8);
#     q=F*Wq;
#     k=F*Wk;
#     %v=F*Wv;
#     Aw=q*k.'./sqrt(8);
#     ff=double(exp(Aw)); 
#     ff=double((ff)); 
#     qq=ff./sum(ff')'; 
#     bb(:,i)=diag(u'*qq*u);

#     fprintf('Aw = %d ',size(Aw));
#     fprintf('\n Wq = %d ',size(Wq));
#     %fprintf('\n f2 = %d',size(f2));
#     fprintf('\n ff = %d',size(ff));
#     fprintf('\n qq = %d',size(qq));
#     fprintf('\n bb = %d',size(bb));

#     %w1=randn(8,1);
#     %w2=randn(8,1);

#     %f=F*W;
#     %f1=f*w1;
#     %f2=f*w2;
#     %ff=f1+f2';
    
#     %fff=max(0,ff)-0.2*max(0,-ff);
    
#     %ff=double(exp(-fff)); 
#     %ff=double((ff)); 
#     %ff=ff.*(A0);   
#     %qq=ff./sum(ff')';    
#     %bb(:,i)=diag(u'*qq*u);
# end
# size(v)

# figure;plot(v(100:end,:),abs(bb(100:end,:)))
# title('Transformer empirical freq response of each simulation on cora');
# xlabel('Eigenvalue');
# ylabel('Magnitude');