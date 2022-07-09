
from nets.TU_graph_classification.SAN_NodeLPE import SAN_NodeLPE
from nets.TU_graph_classification.SAN_EdgeLPE import SAN_EdgeLPE
from nets.TU_graph_classification.SAN import SAN
from nets.TU_graph_classification.SAN_NodeSpectra import SAN_NodeSpectra

def NodeLPE(net_params, edge_features_present=False, node_features_dim=1):
    return SAN_NodeLPE(net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN(net_params)

def NodeSpectra(net_params, edge_features_present=False, node_features_dim=1):
	return SAN_NodeSpectra(net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)

def gnn_model(LPE, net_params, edge_features_present=False, node_features_dim=1):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'none': NoLPE,
        'spectral_node': NodeSpectra
    }
    
    return model[LPE](net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)
