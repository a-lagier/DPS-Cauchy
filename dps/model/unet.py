import torch
import torch.nn as nn
from deepinv.models import DiffUNet

class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = DiffUNet(in_channels, out_channels, large_model=False)

    def load_ckpt(self, checkpoint_file):
        self.model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))

    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model = self.model.to(device)

    def forward(self, x, t):
        return self.model(x, t, type_t="timestep")
