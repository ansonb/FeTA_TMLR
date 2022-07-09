# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import DiffTransformerEncoderLayer
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, ChebConv
import torch.nn.functional as F
from .GenGCN import GENGCN
from .ChebNetDynamic import ChebConvDynamic, ARMAConvDynamic
# from .utils import DEVICE
import numpy as np
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from transformer import utils
from torch_geometric import nn as nng

class GINEPLUS(nng.MessagePassing):
    def __init__(self, fun, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn = fun
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, XX, multihop_edge_index, distance, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(multihop_edge_index[:, distance == i + 1], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

class GraphTransformer(nn.Module):
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

    def forward(self, x, masks, x_pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # we only do mean pooling for now.
        return self.classifier(output)


class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None, return_attn=False):
        output = src
        for mod in self.layers:
            output, attn = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        if return_attn:
            return output, attn
        else:
            return output


class DiffTransformerEncoderGenGCN(nn.TransformerEncoder):
    def __init__(self, d_model, num_heads, *args, num_coefficients=4, laplacian_norm='sym', gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False, use_skip_conn=True, **kwargs):
        super(DiffTransformerEncoderGenGCN, self).__init__(*args, **kwargs)
        self.num_coefficients = num_coefficients
        # if gnn_type=='GENGCN':
        #     self.spectral_gnns = [GENGCN(d_model//num_heads, d_model//num_heads, normalization=laplacian_norm).to(utils.DEVICE) for _ in range(num_heads)]
        # elif gnn_type=='GCNConv':
        #     self.spectral_gnns = [GCNConv(d_model//num_heads, d_model//num_heads).to(utils.DEVICE) for _ in range(num_heads)]
        # elif gnn_type=='ChebConv':
        #     self.spectral_gnns = [ChebConv(d_model//num_heads, d_model//num_heads, num_coefficients, normalization=laplacian_norm).to(utils.DEVICE) for _ in range(num_heads)]
        # elif gnn_type=='ChebConvDynamic':
        #     self.order = self.num_coefficients
        #     self.filter_in_channels = d_model//num_heads
        #     self.filter_out_channels = d_model//num_heads
        #     self.num_coefficients = self.order*self.filter_in_channels*self.filter_out_channels
        #     self.spectral_gnns = [ChebConvDynamic(d_model//num_heads, d_model//num_heads, self.order, normalization=laplacian_norm).to(utils.DEVICE) for _ in range(num_heads)]
        if gnn_type=='GENGCN':
            self.spectral_gnns = GENGCN(d_model//num_heads, d_model//num_heads, normalization=laplacian_norm).to(utils.DEVICE)
        elif gnn_type=='GCNConv':
            self.spectral_gnns = GCNConv(d_model//num_heads, d_model//num_heads).to(utils.DEVICE)
        elif gnn_type=='ChebConv':
            self.spectral_gnns = ChebConv(d_model//num_heads, d_model//num_heads, num_coefficients, normalization=laplacian_norm).to(utils.DEVICE)
        elif gnn_type=='ChebConvDynamic':
            if learn_only_filter_order_coeff:
                self.order = self.num_coefficients
                self.spectral_gnns = ChebConvDynamic(d_model//num_heads, d_model//num_heads, self.num_coefficients, normalization=laplacian_norm, learn_only_filter_order_coeff=True).to(utils.DEVICE)
            else:
                self.order = self.num_coefficients
                self.filter_in_channels = d_model//num_heads
                self.filter_out_channels = d_model//num_heads
                self.num_coefficients = self.order*self.filter_in_channels*self.filter_out_channels
                self.spectral_gnns = ChebConvDynamic(d_model//num_heads, d_model//num_heads, self.order, normalization=laplacian_norm, learn_only_filter_order_coeff=False).to(utils.DEVICE)
        elif gnn_type=='ARMAConvDynamic':
            # if learn_only_filter_order_coeff:
            self.order = self.num_coefficients
            self.num_coefficients = self.num_coefficients*2
            self.spectral_gnns = ARMAConvDynamic(d_model//num_heads, d_model//num_heads, num_stacks=self.order, num_layers=1).to(utils.DEVICE)
            # else:
            #     print('Not Implemented')
        elif gnn_type=='Identity':
            self.spectral_gnns = nn.Identity
        self.gcn = GCNConv(self.num_coefficients, self.num_coefficients)
        self.linear = nn.Linear(self.num_coefficients, self.num_coefficients)
        self.linear_cat = nn.Linear(2*d_model, d_model)

        self.gnn_type = gnn_type
        self.num_heads = num_heads
        self.last_layer_filter = last_layer_filter
        self.learn_only_filter_order_coeff = learn_only_filter_order_coeff
        self.use_skip_conn = use_skip_conn


    def forward(self, src, pe, edge_index, feature_indices, batch, degree=None, mask=None, src_key_padding_mask=None, eigenvalues=None):
        output = src
        num_batches = torch.max(batch)+1
        coefficients = torch.empty((0,num_batches,self.num_coefficients)).to(utils.DEVICE)
        allout_filtered = None 

        import time

        num_layers = len(self.layers)
        for layer_num, mod in enumerate(self.layers):
            t1=time.time()
            output, attn, out_each_head = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask, need_heads=True)
            if 'dynamic' in self.gnn_type.lower():
                if self.last_layer_filter:
                    if layer_num+1 != num_layers:
                        continue
                t2=time.time()
                coeff_all_heads = self.get_filter_coefficients(attn, edge_index, feature_indices, batch, src_key_padding_mask)
                t3=time.time()
                filtered_output_all_heads = []


                coeff = coeff_all_heads.reshape((coeff_all_heads.shape[0]*coeff_all_heads.shape[1],coeff_all_heads.shape[2]))
                out_heads = out_each_head.permute([2,0,1,3]).reshape((out_each_head.shape[2]*out_each_head.shape[0],out_each_head.shape[1],out_each_head.shape[3]))
                batch_size = torch.max(batch)+1
                batch_offset = torch.cat([torch.ones(batch.shape).to(utils.DEVICE)*i*batch_size for i in range(self.num_heads)], dim=0).to(utils.DEVICE)
                batch_all_heads = batch.repeat((self.num_heads)) + batch_offset
                fi_offset = torch.cat([torch.ones((feature_indices.shape[0])).to(utils.DEVICE)*i*batch_size for i in range(self.num_heads)], dim=0).to(utils.DEVICE)
                feature_indices_all_heads = feature_indices.repeat((self.num_heads,1))
                feature_indices_all_heads[:,0] = feature_indices_all_heads[:,0] + fi_offset
                filtered_output_all_heads = self.filter(coeff, out_heads, edge_index, feature_indices_all_heads, batch_all_heads, self.spectral_gnns)
                # for head_idx in range(coeff_all_heads.shape[0]):
                #     coeff = coeff_all_heads[head_idx]
                #     out_cur_head = out_each_head[:,:,head_idx,:]
                #     t40=time.time()
                #     filtered_output_cur_head = self.filter(coeff, out_cur_head, edge_index, feature_indices, batch, self.spectral_gnns[head_idx])
                #     t41=time.time()
                #     # print('= ',t41-t40)
                #     filtered_output_all_heads.append(filtered_output_cur_head)


                t5=time.time()
                coefficients = torch.cat((coefficients, coeff_all_heads), dim=0)
                # filtered_output = torch.cat(filtered_output_all_heads, dim=-1)
                filtered_output = filtered_output_all_heads.reshape((self.num_heads,filtered_output_all_heads.shape[0]//self.num_heads,filtered_output_all_heads.shape[1])).permute([1,0,2]).reshape([filtered_output_all_heads.shape[0]//self.num_heads,filtered_output_all_heads.shape[1]*self.num_heads])
                out_filtered = torch.zeros(output.shape).to(utils.DEVICE)
                out_filtered[feature_indices[:,1], feature_indices[:,0], :] = filtered_output
                # output = output + filtered_output
                # output = output + out_filtered

                # output_cat = torch.cat((output, out_filtered), dim=-1)
                # output = self.linear_cat(output_cat)

                if self.use_skip_conn:
                    if allout_filtered is not None:
                        allout_filtered = allout_filtered + out_filtered
                    else:
                        allout_filtered = out_filtered
                else:
                    allout_filtered = out_filtered
                    output = allout_filtered
            else:
                pass

            # print('== ',t2-t1,t3-t2,t5-t3)
        if self.use_skip_conn:
            if allout_filtered is not None:
                output_cat = torch.cat((output, allout_filtered), dim=-1)
                output = self.linear_cat(output_cat)
                # output = output + allout_filtered
                # output = output
            else:
                pass
        else:
            if allout_filtered is not None:
                output = allout_filtered
            else:
                pass

        if self.norm is not None:
            output = self.norm(output)

        return output, attn, coefficients.permute([1,0,2])

    def get_filter_coefficients(self, attn_weights, edge_index, feature_indices, batch, masks):
        # import time
        # _t0=time.time()
        filter_coeff_all_heads = torch.empty((0,attn_weights.shape[0],self.num_coefficients)).to(utils.DEVICE)
        masks = masks.repeat([attn_weights.shape[1],1])
        inv_masks = ~masks
        g_len_list = torch.sum(inv_masks, dim=-1).detach().cpu().numpy()
        masked_indices = []
        edge_index = []
        batch = []
        node_offset = 0
        # _t1=time.time()
        for b_idx in range(len(inv_masks)):
            g_len = g_len_list[b_idx]
            edge_idx = np.mgrid[node_offset:node_offset+g_len,node_offset:node_offset+g_len]
            edge_index.append(edge_idx.reshape([2,-1]))
            node_offset += g_len
            batch.append(b_idx*np.ones((g_len)))
            masked_indices.append([b_idx*np.ones((g_len)), np.arange(g_len)])
        edge_index = np.concatenate(edge_index, axis=1)
        batch = np.concatenate(batch, axis=0)
        masked_indices = np.concatenate(masked_indices, axis=1).T
        # _t2=time.time()
        edge_index = torch.from_numpy(edge_index).to(utils.DEVICE)
        batch = torch.from_numpy(batch.astype(np.long)).to(utils.DEVICE)
        # _t3=time.time()
        # masked_indices = torch.from_numpy(masked_indices).to(utils.DEVICE)
        inv_masks_int=inv_masks.to(int)
        t1=inv_masks_int.unsqueeze(1).repeat([1,inv_masks.shape[1],1])
        t2=inv_masks_int.unsqueeze(1).repeat([1,inv_masks.shape[1],1]).permute([0,2,1])
        t3=t1*t2
        # edge_weights = attn_weights[:,:,:,:][t3==1]

        # edge_weights = attn_weights[feature_indices[:,0], :, feature_indices[:,1], :][[i for i in range(feature_indices.shape[0])], :, feature_indices[:,1]]

        edge_weight = attn_weights.permute([1,0,2,3]).reshape([self.num_heads*attn_weights.shape[0],attn_weights.shape[2],attn_weights.shape[3]])[t3==1]
        non_zero_indices = torch.where(edge_weight!=0.0)[0].detach()
        # batch_size = torch.max(batch)+1
        # batch_offset = torch.cat([torch.ones(batch.shape).to(utils.DEVICE)*i*batch_size for i in range(self.num_heads)], dim=0).to(utils.DEVICE)
        # batch = batch.repeat((self.num_heads)) + batch_offset
        x_c = torch.ones((attn_weights.shape[0]*self.num_heads, attn_weights.shape[2], self.num_coefficients))[masked_indices[:,0], masked_indices[:,1], :].to(utils.DEVICE)
        edge_weight = edge_weight[non_zero_indices]
        x_c = F.tanh(self.gcn(x_c, edge_index[:,non_zero_indices], edge_weight=edge_weight.detach()))
        gcn_pool = gap(x_c, batch)
        pooled_coeff = self.linear(gcn_pool)
        filter_coeff_all_heads = pooled_coeff.reshape((self.num_heads,attn_weights.shape[0],pooled_coeff.shape[-1]))

        return filter_coeff_all_heads

    # def get_filter_coefficients(self, attn_weights, edge_index, feature_indices, batch, masks):
    #     # import time
    #     _t0=time.time()
    #     filter_coeff_all_heads = torch.empty((0,attn_weights.shape[0],self.num_coefficients)).to(utils.DEVICE)
    #     inv_masks = ~masks
    #     g_len_list = torch.sum(inv_masks, dim=-1).detach().cpu().numpy()
    #     masked_indices = []
    #     edge_index = []
    #     batch = []
    #     node_offset = 0
    #     # _t1=time.time()
    #     for b_idx in range(len(inv_masks)):
    #         g_len = g_len_list[b_idx]
    #         edge_idx = np.mgrid[node_offset:node_offset+g_len,node_offset:node_offset+g_len]
    #         edge_index.append(edge_idx.reshape([2,-1]))
    #         node_offset += g_len
    #         batch.append(b_idx*np.ones((g_len)))
    #         masked_indices.append([b_idx*np.ones((g_len)), np.arange(g_len)])
    #     edge_index = np.concatenate(edge_index, axis=1)
    #     batch = np.concatenate(batch, axis=0)
    #     masked_indices = np.concatenate(masked_indices, axis=1).T
    #     # _t2=time.time()
    #     edge_index = torch.from_numpy(edge_index).to(utils.DEVICE)
    #     batch = torch.from_numpy(batch.astype(np.long)).to(utils.DEVICE)
    #     # _t3=time.time()
    #     # masked_indices = torch.from_numpy(masked_indices).to(utils.DEVICE)
    #     inv_masks_int=inv_masks.to(int)
    #     t1=inv_masks_int.unsqueeze(1).repeat([1,inv_masks.shape[1],1])
    #     t2=inv_masks_int.unsqueeze(1).repeat([1,inv_masks.shape[1],1]).permute([0,2,1])
    #     t3=t1*t2
    #     # edge_weights = attn_weights[:,:,:,:][t3==1]

    #     # edge_weights = attn_weights[feature_indices[:,0], :, feature_indices[:,1], :][[i for i in range(feature_indices.shape[0])], :, feature_indices[:,1]]

    #     for w_idx in range(self.num_heads):
    #         # edge_weight = edge_weights[:,w_idx]
    #         # edge_weight = attn_weights[:,w_idx,:,:][t3==1] + 1e-4 #Unstable
    #         edge_weight = attn_weights[:,w_idx,:,:][t3==1]
    #         non_zero_indices = torch.where(edge_weight!=0.0)[0].detach()

    #         # TODO: use features instead of ones
    #         x_c = torch.ones((attn_weights.shape[0], attn_weights.shape[2], self.num_coefficients))[masked_indices[:,0], masked_indices[:,1], :].to(utils.DEVICE)
    #         # x_c = torch.ones((attn_weights.shape[0], attn_weights.shape[2], self.num_coefficients))[feature_indices[:,0], feature_indices[:,1], :].to(utils.DEVICE)

    #         edge_weight = edge_weight[non_zero_indices]
    #         # x_c = F.relu(self.gcn(x_c, edge_index, edge_weight=edge_weight))

    #         x_c = F.tanh(self.gcn(x_c, edge_index[:,non_zero_indices], edge_weight=edge_weight.detach()))
    #         # x_c = F.tanh(self.gcn(x_c, edge_index[:,non_zero_indices]))

    #         gcn_pool = gap(x_c, batch)
    #         pooled_coeff = self.linear(gcn_pool)
    #         filter_coeff_all_heads = torch.cat([filter_coeff_all_heads, pooled_coeff.unsqueeze(0)], dim=0)
    #     # _t4=time.time()
    #     # print('==== ',_t1-_t0,_t2-_t1,_t3-_t2,_t4-_t3)
    #     return filter_coeff_all_heads

    def filter(self, filter_coeff, graph_signal, edge_index, feature_indices, batch, spectral_gnn, eigenvalues=None):
        x = graph_signal[feature_indices[:,0],feature_indices[:,1],:]
        if self.gnn_type=='GENGCN':
            x_c = F.tanh(spectral_gnn(x, edge_index, batch, filter_coeff))
        elif self.gnn_type=='GCNConv':
            # x_c = F.relu(spectral_gnn(x, edge_index))
            x_c = spectral_gnn(x, edge_index)
        elif self.gnn_type=='ChebConv':
            x_c = F.relu(spectral_gnn(x, edge_index, batch=batch))
        elif self.gnn_type=='ChebConvDynamic':
            if not self.learn_only_filter_order_coeff:
                filter_coeff = filter_coeff.reshape((-1,self.order,self.filter_in_channels, self.filter_out_channels)).permute([1,0,2,3])
            else:
                filter_coeff = filter_coeff.reshape((-1,self.order)).permute([1,0])
            x_c = spectral_gnn(x, edge_index, filter_coeff, batch=batch)
        elif self.gnn_type=='ARMAConvDynamic':
            filter_coeff = filter_coeff.reshape((-1,self.order*2))
            x_c = spectral_gnn(x, edge_index, filter_coeff, batch=batch)
        # TODO: remove pooling
        # pooled_out = gap(x_c, batch)
        pooled_out = x_c

        return pooled_out

    # def get_basis(self, eigenvalues, mask):
    #     Tk = np.zeros([mask.shape[0], self.num_coefficients, mask.shape[1]])
    #     Tk[:,0,:] = 1.
    #     Tk[:,1,:] = eigenvalues
    #     for k in range(2, self.num_coefficients):
    #         Tk[:,k,:] = 2*eigenvalues*Tk[:,k-1,:]  - Tk[:,k-2,:]

    #     return Tk

    # def apply_filter(self, x: Tensor, edge_index: Adj,
    #             batch: Tensor, eigenvalues: List, filter_coeff: Tensor, edge_weight: OptTensor = None) -> Tensor:
    #     """"""
    #     Tk = self.get_basis(eigenvalues)
    #     h = torch.sum(torch.mul(filter_coeff, torch.bmm(Tk, x)), dim=1)

    #     return h

class DiffGraphTransformer(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformer, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, masks, pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # we only do mean pooling for now.
        return self.classifier(output)

class DiffGraphTransformerGCN(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformerGCN, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.gcn = GCNConv(d_model, d_model)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.permute(1, 0, 2)
        output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks, return_attn=True)
        output = output.permute(1, 0, 2)

        x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        x_c = F.relu(self.gcn(x_c, edge_index))
        # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        gcn_pool = gmp(x_c, batch)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)

        output = output + gcn_pool
        # we only do mean pooling for now.
        return self.classifier(output)

class DiffGraphTransformerGenGCN(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0, filter_order=4, gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False):
        super(DiffGraphTransformerGenGCN, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoderGenGCN(d_model, nb_heads, encoder_layer, nb_layers, num_coefficients=filter_order, gnn_type=gnn_type, last_layer_filter=last_layer_filter, learn_only_filter_order_coeff=learn_only_filter_order_coeff)
        self.gcn = GCNConv(d_model, d_model)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0, return_filter_coeff=False):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.permute(1, 0, 2)
        output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn, filter_coeff = self.encoder(output, pe, edge_index, feature_indices, batch, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        output_pooled = self.pooling(output, masks)

        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # gcn_pool = gmp(x_c, batch)
        # output_pooled = output_pooled + gcn_pool

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0

        if return_filter_coeff:
            return self.classifier(output_pooled), filter_coeff_reg, filter_coeff
        else:
            return self.classifier(output_pooled), filter_coeff_reg

    # params: reg_type: type of regularization; options are [pairwise,max]
    def regularisation(self, coeff, reg_type='pairwise'):
        if reg_type=='max':
            gm = torch.bmm(coeff, coeff.permute([0,2,1]))
            mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
            gm = gm*mask
            v1 = torch.norm(coeff,p=2,dim=[2])
            norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
            reg = torch.div(gm,norm_mat)

            # reg = torch.norm(coeff, p=1, dim=[1,2]).mean()
            # if torch.isnan(reg).any():
            #     import pdb; pdb.set_trace()

            # reg = torch.mean(reg)
            reg = torch.max(torch.max(reg, dim=1).values, dim=1).values
            reg = torch.sum(reg)
        elif reg_type=='pairwise':
            gm = torch.bmm(coeff, coeff.permute([0,2,1]))
            mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
            gm = gm*mask
            v1 = torch.norm(coeff,p=2,dim=[2])
            norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
            reg = torch.div(gm,norm_mat)

            reg = torch.norm(coeff, p=2, dim=[1,2]).mean()
            if torch.isnan(reg).any():
                import pdb; pdb.set_trace()
        else:
            print('Regularization not implemented')

        return reg

class GlobalAvg1D(nn.Module):
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)


class DiffGraphTransformerGenGCNMolHiv(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0, filter_order=4, gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False,
                     use_skip_conn=True,
                     use_default_encoder=False):
        super(DiffGraphTransformerGenGCNMolHiv, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        self.d_model = d_model

        # self.embedding = nn.Linear(in_features=in_size,
                                   # out_features=d_model,
                                   # bias=False)
        self.embedding = AtomEncoder(emb_dim = d_model)
        # TODO: use edge embeddings
        self.edge_embeddings = BondEncoder(emb_dim = d_model)

        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        if use_default_encoder:
            self.encoder = DiffTransformerEncoder(d_model, nb_heads, encoder_layer, nb_layers)
        else:
            self.encoder = DiffTransformerEncoderGenGCN(d_model, nb_heads, encoder_layer, nb_layers, num_coefficients=filter_order, gnn_type=gnn_type, last_layer_filter=last_layer_filter, learn_only_filter_order_coeff=learn_only_filter_order_coeff,
            use_skip_conn=use_skip_conn)
        self.gcn = GCNConv(d_model, d_model)
        self.pooling = GlobalAvg1D()
        
        # self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(True),
            nn.Linear(d_model, nb_class)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0, return_filter_coeff=False):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.reshape([-1,x.shape[-1]])
        output = self.embedding(x_t.to(int))

        output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # ##############################################
        # # first layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # output = torch.clone(output)
        # output[feature_indices[:,0],feature_indices[:,1],:] = x_c
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # # gcn_pool = gap(x_c, batch)
        # # output_pooled = gcn_pool
        # ################################################
        # output = output.reshape([x.shape[0],x.shape[1],self.d_model]).permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        # x_t = x.permute(1, 0, 2)
        # output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn, filter_coeff = self.encoder(output, pe, edge_index, feature_indices, batch, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        output_pooled = self.pooling(output, masks)

        # # # x_t = x.permute(1, 0, 2)
        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # gcn_pool = gap(x_c, batch)
        # output_pooled = output_pooled + gcn_pool



        # output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # # # x_t = x.permute(1, 0, 2)
        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # gcn_pool = gap(x_c, batch)
        # output_pooled = gcn_pool
        # # output_pooled = output_pooled + gcn_pool

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0

        cls_out = self.classifier(output_pooled)

        if return_filter_coeff:
            return cls_out.squeeze(), filter_coeff_reg, self.sigmoid(cls_out).squeeze(), filter_coeff
        else:
            return cls_out.squeeze(), filter_coeff_reg, self.sigmoid(cls_out).squeeze()

    def regularisation(self, coeff):
        gm = torch.bmm(coeff, coeff.permute([0,2,1]))
        mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
        gm = gm*mask
        v1 = torch.norm(coeff,p=2,dim=[2])
        norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
        reg = torch.div(gm,norm_mat)

        # reg = torch.norm(coeff, p=1, dim=[1,2]).mean()
        # if torch.isnan(reg).any():
        #     import pdb; pdb.set_trace()

        # reg = torch.mean(reg)
        reg = torch.max(torch.max(reg, dim=1).values, dim=1).values
        reg = torch.sum(reg)
        return reg


class DiffGraphTransformerGenGCNMolPcba(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0, filter_order=4, gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False,
                     use_skip_conn=True,
                     use_default_encoder=False):
        super(DiffGraphTransformerGenGCNMolPcba, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        self.d_model = d_model

        # self.embedding = nn.Linear(in_features=in_size,
                                   # out_features=d_model,
                                   # bias=False)
        self.embedding = AtomEncoder(emb_dim = d_model)
        # TODO: use edge embeddings
        self.edge_embeddings = BondEncoder(emb_dim = d_model)

        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        if use_default_encoder:
            self.encoder = DiffTransformerEncoder(d_model, nb_heads, encoder_layer, nb_layers)
        else:
            self.encoder = DiffTransformerEncoderGenGCN(d_model, nb_heads, encoder_layer, nb_layers, num_coefficients=filter_order, gnn_type=gnn_type, last_layer_filter=last_layer_filter, learn_only_filter_order_coeff=learn_only_filter_order_coeff,
            use_skip_conn=use_skip_conn)
        self.gcn = GCNConv(d_model, d_model)
        self.pooling = GlobalAvg1D()
        
        self.classifier = nn.Linear(in_features=d_model,
                                   out_features=nb_class, bias=True)
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.LeakyReLU(True),
        #     nn.Linear(d_model, nb_class)
        #     )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.reshape([-1,x.shape[-1]])
        output = self.embedding(x_t.to(int))

        output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # ##############################################
        # # first layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # output = torch.clone(output)
        # output[feature_indices[:,0],feature_indices[:,1],:] = x_c
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # # gcn_pool = gap(x_c, batch)
        # # output_pooled = gcn_pool
        # ################################################
        # output = output.reshape([x.shape[0],x.shape[1],self.d_model]).permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        # x_t = x.permute(1, 0, 2)
        # output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn, filter_coeff = self.encoder(output, pe, edge_index, feature_indices, batch, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        output_pooled = self.pooling(output, masks)

        # # # x_t = x.permute(1, 0, 2)
        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # gcn_pool = gap(x_c, batch)
        # output_pooled = output_pooled + gcn_pool



        # output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # # # x_t = x.permute(1, 0, 2)
        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # # gcn_pool = gmp(x_c, batch)
        # gcn_pool = gap(x_c, batch)
        # output_pooled = gcn_pool
        # # output_pooled = output_pooled + gcn_pool

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0

        cls_out = self.classifier(output_pooled)
        return cls_out.squeeze(), filter_coeff_reg, self.sigmoid(cls_out).squeeze()

    def forward_allgcn(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.reshape([-1,x.shape[-1]])
        output = self.embedding(x_t.to(int))

        output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # ##############################################
        # GCN block
        x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        x_c = F.relu(self.gcn(x_c, edge_index))
        x_c = F.relu(self.gcn(x_c, edge_index))
        x_c = F.relu(self.gcn(x_c, edge_index))
        x_c = F.relu(self.gcn(x_c, edge_index))
        x_c = F.relu(self.gcn(x_c, edge_index))
        output = torch.clone(output)
        output[feature_indices[:,0],feature_indices[:,1],:] = x_c
        # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # gcn_pool = gmp(x_c, batch)
        gcn_pool = gap(x_c, batch)
        output_pooled = gcn_pool
        # ################################################

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0

        cls_out = self.classifier(output_pooled)
        return cls_out.squeeze(), filter_coeff_reg, self.sigmoid(cls_out).squeeze()

    def regularisation(self, coeff):
        gm = torch.bmm(coeff, coeff.permute([0,2,1]))
        mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
        gm = gm*mask
        v1 = torch.norm(coeff,p=2,dim=[2])
        norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
        reg = torch.div(gm,norm_mat)

        # reg = torch.norm(coeff, p=1, dim=[1,2]).mean()
        # if torch.isnan(reg).any():
        #     import pdb; pdb.set_trace()

        # reg = torch.mean(reg)
        reg = torch.max(torch.max(reg, dim=1).values, dim=1).values
        reg = torch.sum(reg)
        return reg


class DiffGraphTransformerGenGCNPCQM4M(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0, filter_order=4, gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False):
        super(DiffGraphTransformerGenGCNPCQM4M, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        # self.embedding = nn.Linear(in_features=in_size,
                                   # out_features=d_model,
                                   # bias=False)
        self.embedding = AtomEncoder(emb_dim = d_model)
        # TODO: use edge embeddings
        self.edge_embeddings = BondEncoder(emb_dim = d_model)

        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoderGenGCN(d_model, nb_heads, encoder_layer, nb_layers, num_coefficients=filter_order, gnn_type=gnn_type, last_layer_filter=last_layer_filter, learn_only_filter_order_coeff=learn_only_filter_order_coeff)
        self.gcn = GCNConv(d_model, d_model)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )
        self.d_model = d_model

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        # x_t = x.permute(1, 0, 2)
        # output = self.embedding(x_t.to(torch.long))
        x_t = x.reshape([-1,x.shape[-1]])
        output = self.embedding(x_t.to(int))
        output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        output = output.permute(1, 0, 2).contiguous()
           
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn, filter_coeff = self.encoder(output, pe, edge_index, feature_indices, batch, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        output_pooled = self.pooling(output, masks)

        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # gcn_pool = gmp(x_c, batch)
        # output_pooled = output_pooled + gcn_pool

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0
        return self.classifier(output_pooled), filter_coeff_reg

    def regularisation(self, coeff):
        gm = torch.bmm(coeff, coeff.permute([0,2,1]))
        mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
        gm = gm*mask
        v1 = torch.norm(coeff,p=2,dim=[2])
        norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
        reg = torch.div(gm,norm_mat)

        # reg = torch.norm(coeff, p=1, dim=[1,2]).mean()
        # if torch.isnan(reg).any():
        #     import pdb; pdb.set_trace()

        # reg = torch.mean(reg)
        reg = torch.max(torch.max(reg, dim=1).values, dim=1).values
        reg = torch.sum(reg)
        return reg

class DiffGraphTransformerGenGCNSBM(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0, filter_order=4, gnn_type='ChebConvDynamic', last_layer_filter=True, learn_only_filter_order_coeff=False):
        super(DiffGraphTransformerGenGCNSBM, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoderGenGCN(d_model, nb_heads, encoder_layer, nb_layers, num_coefficients=filter_order, gnn_type=gnn_type, last_layer_filter=last_layer_filter, learn_only_filter_order_coeff=learn_only_filter_order_coeff)
        # self.gcn = GCNConv(d_model, d_model)
        # self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0, return_filter_coeff=False):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.permute(1, 0, 2)
        output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output, attn, filter_coeff = self.encoder(output, pe, edge_index, feature_indices, batch, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        # output_pooled = self.pooling(output, masks)

        # # Last layer GCN block
        # # Comment/Uncomment to use/remove
        # # or better maybe add in args
        # x_c = output[feature_indices[:,0],feature_indices[:,1],:]
        # x_c = F.relu(self.gcn(x_c, edge_index))
        # # gcn_pool = torch.cat([gmp(x_c, batch), gap(x_c, batch)], dim=1)
        # gcn_pool = gmp(x_c, batch)
        # output_pooled = output_pooled + gcn_pool

        if regularization>0:
            filter_coeff_reg = self.regularisation(filter_coeff)
        else:
            filter_coeff_reg = 0

        cls_output = self.classifier(output)
        inv_masks = ~masks
        cls_output = cls_output[inv_masks]

        if return_filter_coeff:
            return cls_output, filter_coeff_reg, filter_coeff
        else:
            return cls_output, filter_coeff_reg

    def regularisation(self, coeff):
        gm = torch.bmm(coeff, coeff.permute([0,2,1]))
        mask = 1.-torch.eye(coeff.shape[1]).unsqueeze(0).repeat([coeff.shape[0],1,1]).to(utils.DEVICE)
        gm = gm*mask
        v1 = torch.norm(coeff,p=2,dim=[2])
        norm_mat = torch.bmm(v1.unsqueeze(-1), v1.unsqueeze(1))
        reg = torch.div(gm,norm_mat)

        # reg = torch.norm(coeff, p=1, dim=[1,2]).mean()
        # if torch.isnan(reg).any():
        #     import pdb; pdb.set_trace()

        # reg = torch.mean(reg)
        reg = torch.max(torch.max(reg, dim=1).values, dim=1).values
        reg = torch.sum(reg)
        return reg

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

class DiffGraphTransformerMolHiv(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformerMolHiv, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        self.d_model = d_model

        # self.embedding = nn.Linear(in_features=in_size,
                                   # out_features=d_model,
                                   # bias=False)
        self.embedding = AtomEncoder(emb_dim = d_model)
        # TODO: use edge embeddings
        self.edge_embeddings = BondEncoder(emb_dim = d_model)

        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        # self.encoder = DiffTransformerEncoder(d_model, nb_heads, encoder_layer, nb_layers)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)

        self.pooling = GlobalAvg1D()
        
        # self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(True),
            nn.Linear(d_model, nb_class)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None, regularization=0.0, return_filter_coeff=False):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.reshape([-1,x.shape[-1]])
        output = self.embedding(x_t.to(int))

        output = output.reshape([x.shape[0],x.shape[1],self.d_model])
        # output = output.reshape([x.shape[0],x.shape[1],self.d_model]).permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        # x_t = x.permute(1, 0, 2)
        # output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)

        # we only do mean pooling for now.
        # we make sure to correctly take the masks into account when pooling
        output_pooled = self.pooling(output, masks)

        cls_out = self.classifier(output_pooled)

        return cls_out.squeeze(), self.sigmoid(cls_out).squeeze()

class DiffGraphTransformerSBM(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformerSBM, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        # self.encoder = DiffTransformerEncoder(d_model, nb_heads, encoder_layer, nb_layers)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        # self.gcn = GCNConv(d_model, d_model)
        # self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, edge_index, batch, feature_indices, masks, pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x_t = x.permute(1, 0, 2)
        output = self.embedding(x_t)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)


        cls_output = self.classifier(output)
        inv_masks = ~masks
        cls_output = cls_output[inv_masks]

        return cls_output

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
