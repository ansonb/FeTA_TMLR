import os
import pickle

import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm


class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(PositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(PositionEncoding):
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        return torch.from_numpy(tmp.toarray())


class AdjEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        return to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)

class FullEncoding(PositionEncoding):
    def __init__(self, savepath, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        try:
            edge_index, edge_attr = get_laplacian(
                        graph.edge_index, edge_attr, normalization=self.normalization, num_nodes=graph.num_nodes)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        if EigVal.shape[0]<graph.num_nodes:
            # import pdb; pdb.set_trace()
            pad_len = graph.num_nodes-EigVal.shape[0]
            EigVal_pad = np.zeros((graph.num_nodes,))
            # EigVec_pad = np.zeros((EigVec.shape[0],graph.num_nodes))
            EigVec_pad = np.zeros((graph.num_nodes,EigVec.shape[1]))
            # TODO: ComplexWarning: Casting complex values to real discards the imaginary part
            EigVal_pad[pad_len:] = EigVal
            # EigVec_pad[:,pad_len:] = EigVec
            EigVec_pad[:-pad_len,:] = EigVec
            EigVec = EigVec_pad
            # start_idx = pad_len+1
            start_idx = 1
        else:
            start_idx = 1
        eig_vec_pe = EigVec[:, start_idx:self.pos_enc_dim+1]
        if eig_vec_pe.shape[1]<self.pos_enc_dim:
            pad_len = self.pos_enc_dim-eig_vec_pe.shape[1]
            eig_vec_pe_pad = np.zeros((eig_vec_pe.shape[0],self.pos_enc_dim))
            eig_vec_pe_pad[:,:eig_vec_pe.shape[1]] = eig_vec_pe
            eig_vec_pe = eig_vec_pe_pad
        # return torch.from_numpy(EigVec[:, start_idx:self.pos_enc_dim+1]).float()
        return torch.from_numpy(eig_vec_pe).float()

    def apply_to(self, dataset):
        dataset.lap_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.lap_pe_list.append(pe)

        return dataset


POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
}
