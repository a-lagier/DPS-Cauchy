import torch
import torch.nn as nn
from deepinv.models import DiffUNet

Tensor = torch.Tensor
Device = torch.device

class UNet(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = DiffUNet(in_channels, out_channels, large_model=False)

    def load_ckpt(self, checkpoint_file: str):
        state_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()

    def eval(self):
        self.model.eval()
    
    def to(self, device: Device):
        self.model = self.model.to(device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t, type_t="timestep")
