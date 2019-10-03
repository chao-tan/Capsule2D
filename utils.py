import torch
from torch import nn


class MMSE(nn.Module):
    def __init__(self):
        super(MMSE,self).__init__()

    def forward(self, inp,oup):
        return torch.mean((inp-oup)*(inp-oup))