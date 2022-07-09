from typing import Optional
from torch_geometric.typing import OptTensor

import torch
from torch.nn import Parameter, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
import math

from typing import Callable
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# from ..inits import glorot, zeros
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class ChebConvDynamic(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True, learn_only_filter_order_coeff=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConvDynamic, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if learn_only_filter_order_coeff:
            self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.learn_only_filter_order_coeff = learn_only_filter_order_coeff

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.learn_only_filter_order_coeff:
            glorot(self.weight)
        zeros(self.bias)


    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(self, x, edge_index, filter_coeff, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        if batch is not None:
            if not self.learn_only_filter_order_coeff:
                _, repeat_indices = torch.unique(batch, sorted=True, return_counts=True)
                weight = torch.repeat_interleave(filter_coeff, repeat_indices, dim=1)
            else:
                _, repeat_indices = torch.unique(batch, sorted=True, return_counts=True)
                filter_coeff = torch.repeat_interleave(filter_coeff, repeat_indices, dim=1)
                weight = self.weight
                # TODO: debug if required
                # weight = self.weight*filter_coeff

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        if self.learn_only_filter_order_coeff:
            out = torch.matmul(filter_coeff[0].unsqueeze(1)*Tx_0, weight[0])
        else:
            out = torch.bmm(Tx_0.unsqueeze(1), weight[0]).squeeze()

        # propagate_type: (x: Tensor, norm: Tensor)
        if weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            if self.learn_only_filter_order_coeff:
                out = out + torch.matmul(filter_coeff[1].unsqueeze(1)*Tx_1, weight[1])
            else:
                out = out + torch.bmm(Tx_1.unsqueeze(1), weight[1]).squeeze()

        for k in range(2, weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            if self.learn_only_filter_order_coeff:
                out = out + torch.matmul(filter_coeff[k].unsqueeze(1)*Tx_2, weight[k])
            else:
                out = out + torch.bmm(Tx_2.unsqueeze(1), weight[k]).squeeze()
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


class ARMAConvDynamic(MessagePassing):
    r"""The ARMA graph convolutional operator from the `"Graph Neural Networks
    with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \frac{1}{K} \sum_{k=1}^K \mathbf{X}_k^{(T)},

    with :math:`\mathbf{X}_k^{(T)}` being recursively defined by

    .. math::
        \mathbf{X}_k^{(t+1)} = \sigma \left( \mathbf{\hat{L}}
        \mathbf{X}_k^{(t)} \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V} \right),

    where :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}` denotes the
    modified Laplacian :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}`.

    Args:
        in_channels (int): Size of each input sample :math:`\mathbf{x}^{(t)}`.
        out_channels (int): Size of each output sample
            :math:`\mathbf{x}^{(t+1)}`.
        num_stacks (int, optional): Number of parallel stacks :math:`K`.
            (default: :obj:`1`).
        num_layers (int, optional): Number of layers :math:`T`.
            (default: :obj:`1`)
        act (callable, optional): Activation function :math:`\sigma`.
            (default: :meth:`torch.nn.ReLU()`)
        shared_weights (int, optional): If set to :obj:`True` the layers in
            each stack will share the same parameters. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the skip connection.
            (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_stacks: int = 1, num_layers: int = 1,
                 shared_weights: bool = False,
                 act: Optional[Callable] = ReLU(), dropout: float = 0.,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ARMAConvDynamic, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act
        self.shared_weights = shared_weights
        self.dropout = dropout

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        T = 1 if self.shared_weights else T

        self.init_weight = Parameter(torch.Tensor(K, F_in, F_out))
        self.weight = Parameter(torch.Tensor(max(1, T - 1), K, F_out, F_out))
        self.root_weight = Parameter(torch.Tensor(T, K, F_in, F_out))

        if bias:
            self.bias = Parameter(torch.Tensor(T, K, 1, F_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.init_weight)
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def _batch_multiply_coeff(self, x, w):
        # x.shape
        # torch.Size([num_nodes, 1, 1, 16])
        # w.shape
        # torch.Size([num_nodes, K, F_in, F_out])
        x1 = x.permute([1,0,2]).unsqueeze(2)
        x1 = x1.repeat([1,w.shape[1],1,1])
        # x1.shape
        # torch.Size([num_nodes, K, 1, F_out])
        x11 = x1.reshape([-1,1,x.shape[-1]])
        w1 = w.reshape([-1,w.shape[-2],w.shape[-2]])
        x2 = torch.bmm(x11,w1)
        # x2.shape
        # torch.Size([num_nodes*K, 1, F_out])
        x21 = x2.reshape([-1,w.shape[1],1,w.shape[-1]])
        x22 = x21.permute([1,0,2,3])
        x23 = x22.reshape([w.shape[1],x.shape[1],x2.shape[-1]])
        # x21.shape
        # torch.Size([num_nodes, K, 1, F_out])
        # x23.shape
        # torch.Size([K, num_nodes, F_out])
        return x23

    def forward(self, x: Tensor, edge_index: Adj, filter_coeff,
                edge_weight: OptTensor = None, batch: OptTensor = None) -> Tensor:
        """"""

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, dtype=x.dtype)

        if batch is not None:
            # TODO: reshape, repeat and bmm
            _, repeat_indices = torch.unique(batch, sorted=True, return_counts=True)
            filter_coeff = torch.repeat_interleave(filter_coeff, repeat_indices, dim=0)
        filter_coeff_a = filter_coeff[:,:self.num_stacks].unsqueeze(-1).unsqueeze(-1)
        filter_coeff_b = filter_coeff[:,self.num_stacks:].unsqueeze(-1).unsqueeze(-1)

        x = x.unsqueeze(-3)
        out = x
        for t in range(self.num_layers):
            if t == 0:
                # out = out @ (self.init_weight * filter_coeff_a)
                w = self.init_weight * filter_coeff_a
                out = self._batch_multiply_coeff(out, w)
            else:
                # out = out @ (self.weight[0 if self.shared_weights else t - 1] * filter_coeff_a)
                w = self.weight[0 if self.shared_weights else t - 1] * filter_coeff_a
                out = self._batch_multiply_coeff(out, w)


            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)

            root = F.dropout(x, p=self.dropout, training=self.training)
            # out += root @ (self.root_weight[0 if self.shared_weights else t] * filter_coeff_b)
            w = self.root_weight[0 if self.shared_weights else t] * filter_coeff_b
            out += self._batch_multiply_coeff(root, w)

            if self.bias is not None:
                out += self.bias[0 if self.shared_weights else t]

            if self.act is not None:
                out = self.act(out)

        return out.mean(dim=-3)


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_stacks={}, num_layers={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_stacks, self.num_layers)
