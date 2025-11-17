import torch
import torch.nn as nn
from deepinv.models import DiffUNet

Tensor = torch.Tensor
Device = torch.device

class UNet(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, large_model: bool = False):
        super().__init__()

        self.model = DiffUNet(in_channels, out_channels, large_model=large_model)
        self.model.eval()
    
    def to(self, device: Device):
        self.model = self.model.to(device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t, type_t="timestep")
