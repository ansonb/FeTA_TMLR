
from nets.ZINC_graph_regression.SAN_NodeLPE import SAN_NodeLPE
from nets.ZINC_graph_regression.SAN_EdgeLPE import SAN_EdgeLPE
from nets.ZINC_graph_regression.SAN import SAN
from nets.ZINC_graph_regression.SAN_NodeSpectra import SAN_NodeSpectra
from nets.ZINC_graph_regression.gat_net import GATNet
from nets.ZINC_graph_regression.gat_feta_net import GATFeTANet


def NodeLPE(net_params):
    return SAN_NodeLPE(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN(net_params)

def NodeSpectra(net_params):
	return SAN_NodeSpectra(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GATFeTA(net_params):
    return GATFeTANet(net_params)
    
def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'none': NoLPE,
        'spectral_node': NodeSpectra,
        'gat': GAT,
        'gat_feta': GATFeTA,
    }
        
    return model[LPE](net_params)