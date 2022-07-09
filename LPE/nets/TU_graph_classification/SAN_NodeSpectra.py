import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.graph_transformer_spectra_layer import GraphTransformerLayerSpectra
from layers.mlp_readout_layer import MLPReadout

class SAN_NodeSpectra(nn.Module):
    def __init__(self, net_params, edge_features_present=False, node_features_dim=1):
        super().__init__()
        
        full_graph = net_params['full_graph']
        self.n_classes = net_params['n_classes']

        gamma = net_params['gamma']
        
        LPE_layers = net_params['LPE_layers']
        LPE_dim = net_params['LPE_dim']
        LPE_n_heads = net_params['LPE_n_heads']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # self.embedding_h = nn.Embedding(node_features_dim, GT_hidden_dim-LPE_dim) #Remove some embedding dimensions to make room for concatenating LPE
        self.embedding_h = nn.Linear(node_features_dim, GT_hidden_dim-LPE_dim, bias=False) #Remove some embedding dimensions to make room for concatenating LPE
        if edge_features_present:
            self.embedding_e_real = BondEncoder(edge_features_dim, GT_hidden_dim)
            self.embedding_e_fake = nn.Embedding(1, GT_hidden_dim)
        self.linear_A = nn.Linear(2, LPE_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)
        
        if not net_params.get('last_layer_filter', False):
            self.layers = nn.ModuleList([ GraphTransformerLayerSpectra(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, filter_order=net_params.get('filter_order',4), edge_features_present=edge_features_present) for _ in range(GT_layers-1) ])
        else:
            self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, edge_features_present=edge_features_present) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayerSpectra(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual, filter_order=net_params.get('filter_order',4), edge_features_present=edge_features_present))
        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)   # 1 out dim for probability      
        
        self.edge_features_present = edge_features_present

    def forward(self, g, h, e, EigVecs, EigVals):
        

        # input embedding
        h = self.embedding_h(h)
        if self.edge_features_present:
            e = self.embedding_e_real(e)
        
        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
        PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
        #1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
        #remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
        #Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        
        #Concatenate learned PE to input embedding
        h = torch.cat((h, PosEnc), 1)

        h = self.in_feat_dropout(h)
          
        # Second Transformer
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
    
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        
        loss = nn.CrossEntropyLoss()
        l = loss(scores.float(), targets)
        
        return l
