from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes

#from ..inits import glorot, zeros
import math
import numpy as np
from torch_sparse import spspmm

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    row, col = indices
    sparse_sizes = x.size()
    return SparseTensor(row = row, col = col,value = values, sparse_sizes = sparse_sizes)

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    ##print("INSIDE GCN NORM")
    ##print("EDGE INDEX >>> ",edge_index)
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        ##print("INSIDE IF")
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
            #print("adjacency >>>>> ",adj_t)
        deg = sum(adj_t, dim=1)
        #print("deg >>>> ",deg)
        ####################
        deg_inv_sqrt = deg.float().pow_(-0.5)
        ###################
        #deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        #print("deg_inv_sqrt >>>>>>>> ", deg_inv_sqrt)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        #print("INSIDE ELSE")
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        #print("DEGREE MATRIX >>> ",deg.shape)
        #print("edge_index >>> ",edge_index.shape)
        #print("edge_weight >> ",edge_weight.shape)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def laplacian_norm(edge_index, batch, edge_weight=None ,num_nodes=None, improved=False,
             add_self_loops=True, dtype=None, num_hops = 4, normalization='sym'):
    if isinstance(edge_index, SparseTensor):
        error("NOT IMPLEMENTED !!!!")
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        batch_np = batch.cpu().numpy()
        indicesValue, indicesList = np.unique(batch_np, return_index=True)

        new_edge_index = torch.empty(2, 0, device=edge_index.device)
        new_edge_weight = torch.empty(0,  device=edge_index.device)
        new_batch = torch.empty(0,  device=edge_index.device)
        hop_index = torch.empty(0, device=edge_index.device)
        for i in range(len(indicesList)):
            start_index = i
            end_index = i+1
            if(end_index == len(indicesList)):
                edge_index_batch = edge_index[:,indicesList[start_index]:]
            else:
                edge_index_batch = edge_index[:,indicesList[start_index]:indicesList[end_index]]

            edge_index_batch_lap, edge_weight_lap = get_laplacian(edge_index_batch, edge_weight=edge_weight, normalization=normalization)
            total_nodes_batch = int(edge_index_batch.max()) + 1 #CHECK IF CAUSING MEMORY ISSUE ELSE MAX - MIN
            #print("new_edge_index >>>>>>>>>>>>> ",new_edge_index)
            #print("edge_index_batch_lap >>>>>>> ",edge_index_batch_lap)
            #print("edge_index_batch >>>>>>>>>>> ",edge_index_batch.shape)
            for hop in range(num_hops):
                #print("hop >>> ",hop)
                if(hop == 0):
                    # new_edge_index = torch.cat((new_edge_index, edge_index_batch_lap), 1)
                    min_node_num = edge_index_batch.min()
                    max_node_num = edge_index_batch.max()
                    self_loop = torch.tensor([[node_idx, node_idx] for node_idx in range(min_node_num,max_node_num+1)]).permute([1,0]).to(edge_index_batch_lap.device)
                    new_edge_index = torch.cat((new_edge_index, self_loop), 1)

                    #print("new_edge_weight >>>>>>>>>>>>>>>>>>>> ", new_edge_weight.shape)
                    #print("111111111s >>>>>>>>>>>>>>>>>>>>>>>>> ", torch.ones(edge_index_batch_lap.size(1), dtype=dtype, device=edge_index_batch_lap.device).shape)
                    new_edge_weight = torch.cat((new_edge_weight, torch.ones(self_loop.size(1), dtype=dtype, device=edge_index_batch_lap.device)), 0)
                    new_batch = torch.cat((new_batch, torch.ones(self_loop.size(1), dtype=dtype, device=edge_index_batch_lap.device)*indicesValue[i]), 0)
                    hop_index = torch.cat((hop_index, torch.ones(self_loop.size(1), dtype=dtype, device=edge_index_batch_lap.device)*hop), 0)
                elif(hop ==1):
                    new_edge_index = torch.cat((new_edge_index, edge_index_batch_lap), 1)
                    new_edge_weight = torch.cat((new_edge_weight, edge_weight_lap), 0)
                    prev_power_edge_index = edge_index_batch_lap
                    prev_power_edge_weigths = edge_weight_lap
                    one_power_edge_index = edge_index_batch_lap
                    one_power_edge_weigths = edge_weight_lap
                    new_batch = torch.cat((new_batch, torch.ones(edge_index_batch_lap.size(1), dtype=dtype, device=edge_index_batch_lap.device)*indicesValue[i]), 0)
                    hop_index = torch.cat((hop_index, torch.ones(edge_index_batch_lap.size(1), dtype=dtype, device=edge_index_batch_lap.device)*hop), 0)
                else:
                    #print("prev_power_edge_index >> ",prev_power_edge_index)
                    #print("prev_power_edge_weigths >>",prev_power_edge_weigths)
                    #print("one_power_edge_index > ",one_power_edge_index)
                    #print("one_power_edge_weigths >> ",one_power_edge_weigths)
                    #print("total_nodes_batch >> ",total_nodes_batch)
                    # index, value = spspmm(prev_power_edge_index, prev_power_edge_weigths, one_power_edge_index, 
                    #                 one_power_edge_weigths, total_nodes_batch, total_nodes_batch, 
                    #                 total_nodes_batch)  #CHECK if they are dense values 
                    sp_prev = torch.sparse_coo_tensor(prev_power_edge_index, prev_power_edge_weigths, (total_nodes_batch, total_nodes_batch))
                    sp_one = torch.sparse_coo_tensor(one_power_edge_index, one_power_edge_weigths, (total_nodes_batch, total_nodes_batch))
                    sp_cur = torch.sparse.mm(sp_prev, sp_one)
                    index = sp_cur._indices()
                    value = sp_cur._values()

                    prev_power_edge_index = index
                    prev_power_edge_weigths = value
                    new_edge_index = torch.cat((new_edge_index, index), 1)
                    new_edge_weight = torch.cat((new_edge_weight, value), 0)
                    new_batch = torch.cat((new_batch, torch.ones(index.size(1), dtype=dtype, device=edge_index_batch_lap.device)*indicesValue[i]), 0)
                    hop_index = torch.cat((hop_index, torch.ones(index.size(1), dtype=dtype, device=edge_index_batch_lap.device)*hop), 0)

        return new_edge_index, new_edge_weight, new_batch, hop_index.to(torch.long)



class GENGCN(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, num_hops=4, normalization='sym', **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GENGCN, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.normalization = normalization

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        torch.nn.init.kaiming_normal(self.weight)

        self.h = Parameter(torch.Tensor(num_hops))
        torch.nn.init.uniform(self.h)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                batch: Tensor, filter_coeff: Tensor, edge_weight: OptTensor = None, eigenvalues=None) -> Tensor:
        """"""
        #convert to sparse here
        ######################################
        ##print("BEFORE >>> ",edge_index)
        #edge_index = to_sparse(edge_index)
        ##print("AFTER >>>  ",edge_index.to_dense())
        #######################################
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, batch, hop_index = laplacian_norm(  # yapf: disable
                        edge_index, batch, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, normalization=self.normalization)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, batch, hop_index)
                else:
                    edge_index, edge_weight, batch, hop_index = cache[0], cache[1], cache[2], cache[3]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, edge_weight, batch, hop_index = laplacian_norm(  # yapf: disable
                        edge_index, batch, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, normalization=self.normalization)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight
        h = self.h.unsqueeze(0).repeat([edge_weight.shape[0],1])
        # _, repeat_indices = torch.unique(batch, sorted=True, return_counts=True)
        # h = torch.repeat_interleave(filter_coeff, repeat_indices, dim=0)
        
        hop_index = hop_index.unsqueeze(1)
        #print("hop_index.shape >>> ",hop_index.shape)
        #print("h.shape >> ",h.shape)
        #print("edge_weight.shape >> ",edge_weight.shape)
        edge_weight = torch.gather(h, dim=-1, index=hop_index).squeeze()*edge_weight
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index.to(torch.long), x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out



    
class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        #convert to sparse here
        ######################################
        ##print("BEFORE >>> ",edge_index)
        #edge_index = to_sparse(edge_index)
        ##print("AFTER >>>  ",edge_index.to_dense())
        #######################################
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)