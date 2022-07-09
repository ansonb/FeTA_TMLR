import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import dgl
import dgl.function as fn
import numpy as np
from scipy import sparse as sp
import math

"""
    GraphiT-GT-LSPE: GraphiT-GT with LSPE
    
"""


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
        # g.edata['ew'] = torch.ones(edge_index.size(1), device=utils.DEVICE)
        g.edata['ew'] = torch.ones(edge_index.size(1), device=g.device)
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
        # l_g = dgl.from_scipy(laplacian_mat, eweight_name='w', device=utils.DEVICE)
        l_g = dgl.from_scipy(laplacian_mat, eweight_name='w', device=g.device)
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


# def src_dot_dst(src_field, dst_field, out_field):
#     def func(edges):
#         return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
#     return func


# def scaling(field, scale_constant):
#     def func(edges):
#         return {field: ((edges.data[field]) / scale_constant)}
#     return func

# Improving implicit attention scores with explicit edge features, if available
# def imp_exp_attn(implicit_attn, explicit_edge):
#     """
#         implicit_attn: the output of K Q
#         explicit_edge: the explicit edge features
#     """
#     def func(edges):
#         return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
#     return func


# def exp(field):
#     def func(edges):
#         # clamp for softmax numerical stability
#         return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
#     return func





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


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

def adaptive_edge_PE(field, adaptive_weight):
    def func(edges):
        # initial shape was: adaptive_weight: [edges,1]; data: [edges, num_heads, 1]
        # repeating adaptive_weight to have: [edges, num_heads, 1]
        edges.data['tmp'] = edges.data[adaptive_weight].repeat(1, edges.data[field].shape[1]).unsqueeze(-1)
        return {'score_soft': edges.data['tmp'] * edges.data[field]}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias, adaptive_edge_PE, attention_for, edge_features_present=True):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph=full_graph
        self.attention_for = attention_for
        self.adaptive_edge_PE = adaptive_edge_PE
        self.edge_features_present = edge_features_present

        if self.attention_for == "h": # attention module for h has input h = [h,p], so 2*in_dim for Q,K,V
            if use_bias:
                self.Q = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                if self.edge_features_present:
                    self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                    if self.edge_features_present:
                        self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                if self.edge_features_present:
                    self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                    if self.edge_features_present:
                        self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
        
        elif self.attention_for == "p": # attention module for p
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                if self.edge_features_present:
                    self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    if self.edge_features_present:
                        self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                if self.edge_features_present:
                    self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    if self.edge_features_present:
                        self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        if self.edge_features_present:
            # Use available edge features to modify the scores for edges
            g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
            
            if self.full_graph:
                g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
    
        g.apply_edges(exp('score'))
        
        # Adaptive weighting with k_RW_eij
        # Only applicable to full graph, For NOW
        if self.adaptive_edge_PE and self.full_graph:
            g.apply_edges(adaptive_edge_PE('score_soft', 'k_RW'))
        # del g.edata['tmp']
        
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
    
    
    def forward(self, g, h, p, e):
        if self.attention_for == "h":
            h = torch.cat((h, p), -1)
        elif self.attention_for == "p":
            h = p
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        if self.edge_features_present:
            E = self.E(e)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            if self.edge_features_present:
                E_2 = self.E_2(e)
            
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        if self.edge_features_present:
            g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            if self.edge_features_present:
                g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        attn = g.edata['score_soft']

        
        del g.ndata['wV']
        del g.ndata['z']
        del g.ndata['Q_h']
        del g.ndata['K_h']
        if self.edge_features_present:
            del g.edata['E']
        
        if self.full_graph:
            del g.ndata['Q_2h']
            del g.ndata['K_2h']
            if self.edge_features_present:
                del g.edata['E_2']
        
        # return h_out
        return h_out, attn
    

class GraphiT_Spectra_LSPE_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, adaptive_edge_PE=False, use_bias=False, filter_order=4, edge_features_present=True):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.edge_features_present = edge_features_present

        self.attention_h = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, adaptive_edge_PE, attention_for="h",edge_features_present=edge_features_present)
        self.attention_p = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, adaptive_edge_PE, attention_for="p",edge_features_present=edge_features_present)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_p = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
        self.mpgnn_layer = GCNLayer(filter_order, filter_order)
        self.filter_order = filter_order
        self.spectral_gnn = ChebLayer(out_dim//num_heads, out_dim//num_heads, filter_order, normalization='sym')
        self.FFN_filter_coeff = nn.Linear(filter_order, filter_order)
        self.mpgnn_layer2 = GCNLayer2(out_dim//num_heads, out_dim//num_heads)
        self.filt_linear = nn.Linear(out_dim//num_heads, out_dim//num_heads)


    def forward(self, g, h, p, e, snorm_n):
        h_in1 = h # for first residual connection
        p_in1 = p # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        
        # multi-head attention out
        # h_attn_out = self.attention_h(g, h, p, e)
        h_attn_out, attn = self.attention_h(g, h, p, e)

        # learn filter coefficients using attention weights
        filter_coeff, g_all_heads = self.get_filter_coeff(g, attn)
        g_all_heads.ndata['h_attn'] = h_attn_out.permute([1,0,2]).reshape([-1,h_attn_out.shape[-1]])
        batch = g_all_heads.batch_num_nodes()
        g.ndata['x'] = h_attn_out
        h_filt_out = self.filter(g_all_heads, filter_coeff, batch, feature_name='h_attn') # [num_nodes*num_heads,filter_order]
        h_filt_out = self.filt_linear(torch.tanh(h_filt_out))

        num_nodes = g.num_nodes()
        h_filt_out = h_filt_out.reshape([self.num_heads,num_nodes,self.out_channels//self.num_heads]).permute([1,0,2]).reshape((num_nodes,self.out_channels))

        #Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)
        h = h + h_filt_out

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection
            
        # # GN from benchmarking-gnns-v1
        # h = h * snorm_n

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        # # GN from benchmarking-gnns-v1
        # h = h * snorm_n
        
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         
        
        # [END] For calculation of h -----------------------------------------------------------------
        
        
        # [START] For calculation of p -----------------------------------------------------------------
        
        # multi-head attention out
        p_attn_out, p_attn = self.attention_p(g, None, p, e)

        # learn filter coefficients using attention weights
        p_filter_coeff, p_g_all_heads = self.get_filter_coeff(g, attn)
        p_g_all_heads.ndata['p_attn'] = p_attn_out.permute([1,0,2]).reshape([-1,p_attn_out.shape[-1]])
        batch = p_g_all_heads.batch_num_nodes()
        g.ndata['xp'] = p_attn_out
        p_filt_out = self.filter(p_g_all_heads, p_filter_coeff, batch, feature_name='p_attn') # [num_nodes*num_heads,filter_order]
        p_filt_out = self.filt_linear(torch.tanh(p_filt_out))

        num_nodes = g.num_nodes()
        p_filt_out = p_filt_out.reshape([self.num_heads,num_nodes,self.out_channels//self.num_heads]).permute([1,0,2]).reshape((num_nodes,self.out_channels))

        #Concat multi-head outputs
        p = p_attn_out.view(-1, self.out_channels)
        p = p + p_filt_out

        #Concat multi-head outputs
        p = p_attn_out.view(-1, self.out_channels)
       
        p = F.dropout(p, self.dropout, training=self.training)

        p = self.O_p(p)
        
        p = torch.tanh(p)
        
        if self.residual:
            p = p_in1 + p # residual connection

        # [END] For calculation of p -----------------------------------------------------------------

        return h, p
       
    def get_filter_coeff(self, g, attn_w):

        attn = attn_w.permute([1,0,2]).reshape([-1,attn_w.shape[2]])
        all_graphs = dgl.unbatch(g)
        graphs_list = []
        for _ in range(self.num_heads):
            graphs_list.extend(all_graphs.copy())
        new_g = dgl.batch(graphs_list)
        num_nodes = new_g.num_nodes()

        new_g.edata['ew'] = attn.detach()
        new_g.ndata['x'] = torch.ones((num_nodes,self.filter_order,)).to(new_g.device)

        # x_c = F.tanh(self.mpgnn_layer(new_g, torch.ones((self.filter_order,)).to(utils.DEVICE), edge_weight=attn.detach()))
        # x_c = F.tanh(self.mpgnn_layer(new_g).to(utils.DEVICE))
        x_c = F.tanh(self.mpgnn_layer(new_g).to(new_g.device))
        new_g.ndata['h'] = x_c
        h_g = dgl.mean_nodes(new_g, 'h')
        pooled_coeff = self.FFN_filter_coeff(h_g)

        filter_coeff_all_heads = pooled_coeff.reshape((self.num_heads,len(all_graphs),pooled_coeff.shape[-1]))

        return filter_coeff_all_heads, new_g

    def filter(self, g, filter_coeff, batch, feature_name='h_attn'):
        filter_coeff = filter_coeff.reshape((-1,self.filter_order)).permute([1,0])
        h = self.spectral_gnn(g, filter_coeff, lambda_max=None, batch=batch, feature_name=feature_name)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)