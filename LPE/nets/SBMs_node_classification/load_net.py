"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.SAN_NodeLPE import SAN_NodeLPE
from nets.SBMs_node_classification.SAN_EdgeLPE import SAN_EdgeLPE
from nets.SBMs_node_classification.SAN import SAN
from nets.SBMs_node_classification.SAN_NodeSpectra import SAN_NodeSpectra


def NodeLPE(net_params):
    return SAN_NodeLPE(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN(net_params)

def NodeSpectra(net_params):
    return SAN_NodeSpectra(net_params)
    
def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'none': NoLPE,
        'spectral_node': NodeSpectra
    }
        
    return model[LPE](net_params)