import torch

def init_device():
	global DEVICE
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
