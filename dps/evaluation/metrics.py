import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import lpips

from ..utils import clip_to_pixel

Tensor = torch.Tensor
Device = torch.device

__METRIC__ = {}

def register_metric(name: str):
    def wrapper(cls):
        if __METRIC__.get(name, None):
            raise NameError(f"{name} already registerd")
        __METRIC__[name] = cls
        return cls
    return wrapper

def get_metric(name, **kwargs):
    if isinstance(name, list):
        return MetricWrapper([__METRIC__[n](**kwargs) for n in name])
    else:
        if __METRIC__.get(name, None) is None:
            raise NameError(f"Name {name} is not defined.")
        return __METRIC__[name](**kwargs)

class MetricWrapper():
    def __init__(self, metrics):
        self.metrics = metrics

    def get_names(self) -> list:
        return sum([m.get_names() for m in self.metrics], [])

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        return {k: v for m in self.metrics for k,v in m.eval(x,y).items()}

class Metric():

    def __init__(self):
        pass
    
    def get_names(self) -> list:
        return []
    
    def prepare_data(self, x: Tensor, y: Tensor, to_numpy=True) -> tuple:
        assert x.size(0) == y.size(0)

        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        x = x.detach().cpu()
        y = y.detach().cpu()

        if to_numpy:
            x = x.numpy()
            y = y.numpy()

        return x, y

    def eval(self, x: Tensor, y: Tensor) -> dict:
        x, y = self.prepare_data(x, y)

        return {'mse': np.abs(x - y).mean(-1)}

@register_metric('psnr')
class PSNR(Metric):

    def __init__(self, **kwargs):
        super().__init__()

    def get_names(self) -> list:
        return ['min_psnr', 'avg_psnr', 'max_psnr']
    
    def eval(self, x: Tensor, y: Tensor) -> dict:
        x, y = self.prepare_data(x, y)

        x = clip_to_pixel(x)
        y = clip_to_pixel(y)

        mse = np.power(x - y, 2).mean(-1)
        max_ = x.max(-1)

        psnr = 20 * np.log10(max_) - 10 * np.log10(mse)

        return {'min_psnr': psnr.min(),
                'avg_psnr': psnr.mean(),
                'max_psnr': psnr.max()}

@register_metric('fid')
class FID(Metric):

    def __init__(self, **kwargs):
        super().__init__()

        self.inception = inception_v3(transform_input=False, weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()
    
    def get_names(self) -> list:
        return ['fid']

    def eval(self, x: Tensor, y: Tensor) -> dict:
        y = y.to(x.device)
        self.inception.to(x.device)

        x = self.inception(x)
        y = self.inception(y)

        x, y = self.prepare_data(x, y)

        mu_x = x.mean(0)
        mu_y = y.mean(0)
        mu_mse = np.power(mu_x - mu_y, 2).sum()

        sigma_x = np.cov(x, rowvar=False)
        sigma_y = np.cov(y, rowvar=False)

        return {'fid': mu_mse + np.trace(sigma_x + sigma_y + 2 * sqrtm(sigma_x @ sigma_y).real)}


@register_metric('lpips')
class LPIPS(Metric):

    def __init__(self, **kwargs):
        super().__init__()

        self.lpips = lpips.LPIPS(net='vgg')
    
    def get_names(self) -> list:
        return ['lpips']
    
    def eval(self, x: Tensor, y: Tensor) -> dict:
        x = x.detach().cpu()
        y = y.detach().cpu()

        return {'lpips': self.lpips(x, y).mean().item()}