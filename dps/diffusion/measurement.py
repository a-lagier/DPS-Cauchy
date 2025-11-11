import torch
import torch.nn as nn
import torch.nn.functional as F

from motionblur.motionblur import Kernel

from ..utils import make_mask, img_range, clip_to_noise, Blurkernel, normalize

Tensor = torch.Tensor
Device = torch.device

__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"{name} already registerd")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __NOISE__[name](**kwargs)

class Noise():

    def __init__(self):
        self.sampler = None
        return
    
    def sample(self, size: tuple) -> Tensor: # see distribution.sample_n
        return self.sampler.sample(sample_shape=size)
    
    def apply(self, x: Tensor) -> Tensor:
        return x + self.sample(x.shape).to(x.device)
    
    def log_likelihood(self, target: Tensor, x: Tensor) -> Tensor:
        return target - x

@register_noise('clean')
class Clean(Noise):

    def __init__(self, **kwargs):
        super().__init__()
    
    def apply(self, x: Tensor) -> Tensor:
        return x
    
    def log_likelihood(self, target: Tensor, x: Tensor) -> Tensor:
        return torch.linalg.norm(target - x)

@register_noise('gaussian')
class Gaussian(Noise):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.mean = kwargs.get('mean', 0.0)
        self.std = kwargs.get('std', 1.0)
        self.sampler = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)
    
    def log_likelihood(self, target, x):
        return torch.linalg.norm(target - x)

@register_noise('poisson')
class Poisson(Noise):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.rate = kwargs.get('rate', 1.0)
        self.sampler = torch.distributions.poisson.Poisson(rate=self.rate)
    
    def apply(self, x: Tensor) -> Tensor:
        # TODO : clean the noisy poisson implementation
        pixel_x = ((x + 1.0) * 255 / 2).detach()
        pixel_out = pixel_x + self.sample(pixel_x.shape).to(x.device)
        pixel_out = torch.clamp(pixel_out, 0, 255)
        out = clip_to_noise(pixel_out.float() / 255.0)
        return out

    def log_likelihood(self, target: Tensor, x: Tensor) -> Tensor:
        z = torch.linalg.norm(target - x) / target.abs()
        return z.mean()

@register_noise('binomial')
class Binomial(Noise):

    def __init__(self, **kwargs):
        super().__init__()
        n = kwargs.get('n', 1)
        probs = kwargs.get('probs', 0.5)
        self.sampler = torch.distributions.binomial.Binomial(total_count=n, probs=probs)

    def log_likelihood(self, target: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()

@register_noise('gamma')
class Gamma(Noise):
    def __init__(self, **kwargs):
        super().__init__()
        self.concentration = kwargs.get('concentration', 0.5)
        self.rate = kwargs.get('rate', 1.0)
        self.sampler = torch.distributions.gamma.Gamma(concentration=self.concentration, rate=self.rate)
    
    def log_likelihood(self, target: Tensor, x: Tensor) -> Tensor:
        z = target - x
        z_safe = torch.clamp(z, min=1e-8)

        log_prob = (
            (self.concentration - 1) * z * torch.log(z_safe)
            - self.rate * z
        )
        return log_prob.mean()

        alpha = self.concentration - 1
        lamb = self.rate
        def f(x):
            return (x**2 / ((x/alpha) ** 2 + 1)) + lamb * x ** 2
        # norm = torch.linalg.norm(target - x)
        norm = f(target - x).mean()
        # norm = alpha * (target - x) * (target - x).log() - lamb * (target - x)
        # print(norm.mean().item())
        return norm

@register_noise('cauchy')
class Cauchy(Noise):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.1)
        self.location = kwargs.get('location', 0.0)
        self.sampler = torch.distributions.cauchy.Cauchy(self.location, self.scale)
    
    def apply(self, x:Tensor) -> Tensor:
        pixel_x = ((x + 1.0) * 255 / 2)
        pixel_out = pixel_x + self.sample(pixel_x.shape).to(x.device)
        pixel_out = torch.clamp(pixel_out, 0, 255)
        out = clip_to_noise(pixel_out.float() / 255.0)
        return out

    def log_likelihood(self, target: Tensor, x:Tensor) -> Tensor:
        return torch.linalg.norm(target - x) / self.scale ** 2


__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"{name} already registerd")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class Operator():

    def __init__(self):
        return

    def transform(self, x: Tensor) -> Tensor:
        return x
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        return x
    
    def grad_compute(self, x_next: Tensor, x_prev: Tensor, approx_x: Tensor, y: Tensor, noise: Noise, scale: float = 1.0, **kwargs) -> Tensor:
        Ax = self.transform(approx_x)
        data_fidelity = noise.log_likelihood(y, Ax)
        grad_ = torch.autograd.grad(outputs=data_fidelity, inputs=x_prev)[0]
        # grad_ = torch.autograd.grad(outputs=data_fidelity, inputs=approx_x)[0]
        grad_ = grad_.detach()
        data_fidelity = data_fidelity.detach()
        # print("y", img_range(y))
        # print("Ax", img_range(Ax))
        # print("norm", data_fidelity.item())
        # print("grad_norm", img_range(grad_))

        # line 6 of algorithm 1
        # step = kwargs.get('step', -1)
        # if step >= 900:
        #     return x_next, data_fidelity
        x_next -= scale * grad_
        return x_next, data_fidelity

@register_operator('identity')
class Identity(Operator):

    def __init__(self, device: Device, **kwargs):
        super().__init__()

    def transform(self, x: Tensor) -> Tensor:
        return x

@register_operator('super-resolution')
class SuperResolutionOperator(Operator):

    def __init__(self, device: Device, **kwargs):
        super().__init__()

        self.device = device
        self.resolution_factor = kwargs.get('resolution_factor', 0.5)

    def transform(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=self.resolution_factor, mode='nearest') # change to find linear operator
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=1./self.resolution_factor, mode="nearest")


@register_operator('inpainting')
class Inpainting(Operator):

    def __init__(self, device: Device, **kwargs):
        super().__init__()

        self.device = device
        self.set_mask(**kwargs)
    
    def set_mask(self, **kwargs):
        image_size = kwargs.get('sample_size', None)
        mask_type = kwargs.get('mask_type', 'box')
        mask_size = kwargs.get('mask_size', 10)
        mask_density = kwargs.get('mask_density', 0.2)

        self.mask = make_mask(image_size, mask_type, (mask_size, mask_size), mask_density).to(self.device)

    def transform(self, x: Tensor) -> Tensor:
        return x * self.mask
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        return super().inverse_transform(x)



# This operator class was taken from the original code
@register_operator('motion-blur')
class MotionBlur(Operator):
    def __init__(self, device: Device, **kwargs):
        self.device = device

        kernel_size = kwargs.get('kernel_size', 3)
        self.kernel_size = kernel_size
        intensity = kwargs.get('intensity', 1.0)
        
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def transform(self, data: Tensor) -> Tensor:
        return self.conv(data)

    def inverse_transform(self, data: Tensor) -> Tensor:
        return data

    def get_kernel(self) -> Tensor:
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


# This operator class was taken from the original code
@register_operator('gaussian-blur')
class GaussialBlur(Operator):
    def __init__(self, device: Device, **kwargs):
        self.device = device
        
        kernel_size = kwargs.get('kernel_size', 3)
        self.kernel_size = kernel_size
        intensity = kwargs.get('intensity', 1.0)

        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def transform(self, data: Tensor) -> Tensor:
        return self.conv(data)

    def inverse_transform(self, data: Tensor) -> Tensor:
        return data

    def get_kernel(self) -> Tensor:
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator('phase-retrieval')
class PhaseRetrieval(Operator):

    def __init__(self, device: Device, **kwargs):
        super().__init__()
        self.device = device
        
        oversample = kwargs.get("oversample", 2.0)
        self.pad_dim = int(oversample / 8.0 * 256)

    def transform(self, data: Tensor) -> Tensor:
        data = F.pad(data, (self.pad_dim, self.pad_dim, self.pad_dim, self.pad_dim))

        if not torch.is_complex(data):
            data = data.type(torch.complex64)
    
        data = torch.view_as_real(data)
        data = torch.fft.ifftshift(data, dim=[-3, -2])
        data = torch.view_as_complex(data)
        data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
        data = torch.view_as_real(data)
        data = torch.fft.fftshift(data, dim=[-2, -1])
        data = torch.view_as_complex(data)
        data = data.abs()

        # data = normalize(data)

        return data
    
    def inverse_transform(self, data: Tensor) -> Tensor:
        return data

# # This operator class was taken from the original code
# @register_operator(name='nonlinear-blur')
# class NonlinearBlurOperator(Operator):
#     def __init__(self, opt_yml_path, device):
#         self.device = device
#         self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
#     def prepare_nonlinear_blur_model(self, opt_yml_path):
#         '''
#         Nonlinear deblur requires external codes (bkse).
#         '''
#         from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

#         with open(opt_yml_path, "r") as f:
#             opt = yaml.safe_load(f)["KernelWizard"]
#             model_path = opt["pretrained"]
#         blur_model = KernelWizard(opt)
#         blur_model.eval()
#         blur_model.load_state_dict(torch.load(model_path)) 
#         blur_model = blur_model.to(self.device)
#         return blur_model
    
#     def forward(self, data, **kwargs):
#         random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
#         data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
#         blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
#         blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
#         return blurred
