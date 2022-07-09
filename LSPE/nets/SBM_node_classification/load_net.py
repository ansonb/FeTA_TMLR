"""
    Utility file to select GraphNN model as
    selected by the user
"""

# from nets.ZINC_graph_regression.gatedgcn_net import GatedGCNNet
# from nets.ZINC_graph_regression.pna_net import PNANet
# from nets.ZINC_graph_regression.san_net import SANNet
from nets.SBM_node_classification.graphit_net import GraphiTNet
from nets.SBM_node_classification.graphit_spectra_net import GraphiTSpectraNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def PNA(net_params):
    return PNANet(net_params)

def SAN(net_params):
    return SANNet(net_params)

def GraphiT(net_params, edge_features_present=False, node_features_dim=1):
    return GraphiTNet(net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)

def Spectra(net_params, edge_features_present=False, node_features_dim=1):
    return GraphiTSpectraNet(net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)

def gnn_model(MODEL_NAME, net_params, edge_features_present=False, node_features_dim=1):
    models = {
        'GatedGCN': GatedGCN,
        'PNA': PNA,
        'SAN': SAN,
        'GraphiT': GraphiT,
        'Spectra': Spectra
    }
        
    return models[MODEL_NAME](net_params, edge_features_present=edge_features_present, node_features_dim=node_features_dim)