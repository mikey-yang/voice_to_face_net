import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

CROSSENTROPY = CrossEntropyLoss()

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return CROSSENTROPY(output, target)
