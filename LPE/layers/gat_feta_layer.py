import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
import dgl
import dgl.function as fn
import numpy as np
from scipy import sparse as sp
from torch.nn import Parameter

import utils
import math


"""
    Util functions
"""

###### MPGNN to learn filter coefficeints ############
# gcn_msg = fn.copy_src(src='h', out='m')
def gcn_msg(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges._edge_data[dst_field])}
    return func
# self.src['x']*self._edge_data['ew']
def send_source(edges): return {'m': edges.src['x']*edges._edge_data['ew']}
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, edge_weights=None):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'ew'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            if edge_weights is not None:
                g.edata['ew'] = edge_weights
            # g.update_all(gcn_msg, gcn_reduce)
            # g.apply_edges(gcn_msg('h', 'ew', 'h_out'), edges=real_ids)
            g.update_all(send_source, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

###### MPGNN to learn filter coefficeints ############
gcn_msg2 = fn.copy_src(src='x', out='m')
gcn_reduce2 = fn.sum(msg='m', out='h')
class GCNLayer2(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer2, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'ew'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.update_all(gcn_msg2, gcn_reduce2)
            h = g.ndata['h']
            return self.linear(h)


###### ChebNet based filter ############
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def cheb_msg(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges._edge_data[dst_field])}
    return func
def cheb_msg2(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges._edge_data[dst_field].unsqueeze(1))}
    return func
def cheb_reduce(src_field, out_field):
    # def func(msg=src_field, out=out_field):
    #     return fn.sum(msg=src_field, out=out_field)
    # return func
    return fn.sum(msg=src_field, out=out_field)
# use laplacian function from chebnet
def get_laplacian(g, norm='sym'):
    g = dgl.remove_self_loop(g)
    num_nodes = g.number_of_nodes()
    if g.edata.get('ew') is None:
        edge_index = g.get_edges()
        g.edata['ew'] = torch.ones(edge_index.size(1), device=utils.DEVICE)
    row, col = g.get_edges()
    deg = g.in_degree() + g.out_degree()
    if norm is None:
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        # g = dgl.add_self_loop(g)
        # laplacian = sparse.eye(num_nodes) - adj
        # TODO: add edge weights to the adj
        laplacian = deg*sparse.eye(num_nodes) - adj
    elif norm=='sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        ew = deg_inv_sqrt[row] * adj * deg_inv_sqrt[col]
        laplacian = sparse.eye(num_nodes) - ew
    elif norm=='rw':
        deg_inv = deg.pow_(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        ew = deg_inv[row] * adj
        laplacian = sparse.eye(num_nodes) - ew

    return laplacian

def get_laplacian_sp(g, norm='sym'):
    n = g.number_of_nodes()
    # TODO: replace with DGLGraph.adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    return L

class ChebLayer(nn.Module):
    def __init__(self, in_feats, out_feats, K, 
        normalization='sym', bias=True, **kwargs):
        super(ChebLayer, self).__init__()
        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_feats
        self.out_channels = out_feats
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_feats, out_feats))

        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, g, filter_coeff, batch=None, lambda_max=None, feature_name='h_attn'):


        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        if lambda_max is None:
            lambda_max = 2.0


        if batch is not None:
            # _, repeat_indices = torch.unique(batch, sorted=True, return_counts=True)
            filter_coeff = torch.repeat_interleave(filter_coeff, batch, dim=1)
        weight = self.weight

        laplacian_mat = get_laplacian_sp(g)
        laplacian_mat = 2.*laplacian_mat/lambda_max - sp.eye(g.number_of_nodes())
        laplacian_mat = laplacian_mat.astype(np.float32)
        l_g = dgl.from_scipy(laplacian_mat, eweight_name='w', device=utils.DEVICE)
        if torch.isnan(l_g.edata['w']).any():
            import pdb; pdb.set_trace()
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'Tx_is'` ndata below) are automatically popped out
        # when the scope exits.
        with l_g.local_scope():
            l_g.ndata[feature_name] = g.ndata[feature_name]
            l_g.ndata['Tx_0'] = g.ndata[feature_name]
            l_g.ndata['Tx_1'] = g.ndata[feature_name]  # Dummy.
            out = torch.matmul(filter_coeff[0].unsqueeze(1)*l_g.ndata['Tx_0'], weight[0])

            # propagate_type: (x: Tensor, norm: Tensor)
            if weight.size(0) > 1:
                # Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
                l_g.update_all(cheb_msg2(feature_name,'w','Tx_1_p'), cheb_reduce('Tx_1_p','Tx_1'))
                # g.edata['w'] stores the edge weights
                # l_g.update_all(fn.u_mul_e(feature_name, 'w', 'm'), fn.sum('m', 'Tx_1'))
                out = out + torch.matmul(filter_coeff[1].unsqueeze(1)*l_g.ndata['Tx_1'], weight[1])

            for k in range(2, weight.size(0)):
                # Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
                l_g.update_all(cheb_msg2('Tx_1','w','Tx_2_p'), cheb_reduce('Tx_2_p','Tx_2'))
                # g.edata['w'] stores the edge weights
                # l_g.update_all(fn.u_mul_e('Tx_1', 'w', 'm'), fn.sum('m', 'Tx_2'))
                l_g.ndata['Tx_2'] = 2. * l_g.ndata['Tx_2'] - l_g.ndata['Tx_0']
                out = out + torch.matmul(filter_coeff[k].unsqueeze(1)*l_g.ndata['Tx_2'], weight[k])
                l_g.ndata['Tx_0'] = l_g.ndata['Tx_1']
                l_g.ndata['Tx_1'] = l_g.ndata['Tx_2']

            if self.bias is not None:
                out += self.bias

        return out


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func


def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L*torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


"""
    GAT + FeTA: Frequency Transformer Attention
"""

class GATFeTALayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
    filter_order : 
        Order of the filter or the number of filter coefficient (basis) to be used by the filter.
        Default: 4
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu, filter_order=4):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

        self.mpgnn_layer = GCNLayer(filter_order, filter_order)
        # self.spectral_gnn = ChebLayer(out_dim//num_heads, out_dim//num_heads, filter_order, normalization='sym')
        # Here out_dim is the output dimension of each head and not the final output dimension after concat
        self.spectral_gnn = ChebLayer(out_dim, out_dim, filter_order, normalization='sym')
        self.FFN_filter_coeff = nn.Linear(filter_order, filter_order)
        self.out_channels = out_dim*num_heads
        # self.filt_linear = nn.Linear(out_dim//num_heads, out_dim//num_heads)
        self.filt_linear = nn.Linear(out_dim, out_dim)
        self.filter_order = filter_order
        self.num_heads = num_heads

    def forward(self, g, h):

        # import pdb; pdb.set_trace()

        h_in = h # for residual connection

        # h = self.gatconv(g, h).flatten(1)
        h, attn = self.gatconv(g, h, get_attention=True)
        h_attn_out = h

        # learn filter coefficients using attention weights
        filter_coeff, g_all_heads = self.get_filter_coeff(g, attn)
        g_all_heads.ndata['h_attn'] = h_attn_out.permute([1,0,2]).reshape([-1,h_attn_out.shape[-1]])
        batch = g_all_heads.batch_num_nodes()
        g.ndata['x'] = h_attn_out
        h_filt_out = self.filter(g_all_heads, filter_coeff, batch, feature_name='h_attn') # [num_nodes*num_heads,filter_order]
        h_filt_out = self.filt_linear(torch.tanh(h_filt_out))
        num_nodes = g.num_nodes()
        h_filt_out = h_filt_out.reshape([self.num_heads,num_nodes,self.out_channels//self.num_heads]).permute([1,0,2]).reshape((num_nodes,self.out_channels))

        h = h.flatten(1)
        h = h + h_filt_out

        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h
    
    def get_filter_coeff(self, g, attn_w):

        # attn = attn_w.permute([1,0,2]).reshape([-1,attn_w.shape[2]])
        attn = attn_w.permute([1,0,2]).reshape([-1,attn_w.shape[2]])
        all_graphs = dgl.unbatch(g)
        graphs_list = []
        for _ in range(self.num_heads):
            graphs_list.extend(all_graphs.copy())
        new_g = dgl.batch(graphs_list)
        num_nodes = new_g.num_nodes()
        new_g.edata['ew'] = attn.detach()
        new_g.ndata['x'] = torch.ones((num_nodes,self.filter_order,)).to(new_g.device)

        x_c = F.tanh(self.mpgnn_layer(new_g).to(utils.DEVICE))
        new_g.ndata['h'] = x_c
        h_g = dgl.mean_nodes(new_g, 'h')
        pooled_coeff = self.FFN_filter_coeff(h_g)

        filter_coeff_all_heads = pooled_coeff.reshape((self.num_heads,len(all_graphs),pooled_coeff.shape[-1]))

        return filter_coeff_all_heads, new_g

    def filter(self, g, filter_coeff, batch, feature_name='h_attn'):
        filter_coeff = filter_coeff.reshape((-1,self.filter_order)).permute([1,0])
        h = self.spectral_gnn(g, filter_coeff, lambda_max=None, batch=batch, feature_name=feature_name)
        return h


    
