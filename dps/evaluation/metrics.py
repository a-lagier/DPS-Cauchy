import torch
import numpy as np

__METRIC__ = {}

def register_metric(name):
    def wrapper(cls):
        if __METRIC__.get(name, None):
            raise NameError(f"{name} already registerd")
        __METRIC__[name] = cls
        return cls
    return wrapper

def get_metric(name: str, **kwargs):
    if __METRIC__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __METRIC__[name](**kwargs)

class Metric():

    def __init(self):
        pass
    
    def get_names(self):
        return []
    
    def prepare_data(self, x, y):

        print('prepare data')
        print(x.shape)
        print(y.shape)

        assert x.size(0) == y.size(0)
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        return x, y

    def eval(self, x, y):
        x, y = self.prepare_data(x, y)

        return np.abs(x - y).mean(-1)

@register_metric('psnr')
class PSNR(Metric):

    def __init__(self, **kwargs):
        super().__init__()

    def get_names(self):
        return ['min_psnr', 'avg_psnr', 'max_psnr']
    
    def eval(self, x, y):
        x, y = self.prepare_data(x, y)

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
    
    def get_names(self):
        return ['fid']

    def eval(self, x, y):
        x, y = self.prepare_data(x, y)

        return