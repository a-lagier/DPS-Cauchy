import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

from motionblur.motionblur import Kernel


def load_image(filename):
    array = plt.imread(filename)
    x = torch.from_numpy(array).float()
    x = x[:,:,:3].permute(2, 0, 1)
    x = normalize(x)
    x = x * 2 - 1
    return x

def prepare_image(img):
    if img.ndim == 4 and img.size(0) > 1:
        img = img[0]

    img = img.detach().cpu().squeeze()
    img = torch.clamp((img + 1) / 2, 0, 1).numpy()
    return np.transpose(img, (1, 2, 0))

def normalize(image_tensor):
    image_tensor -= image_tensor.min()
    image_tensor /= image_tensor.max()
    return image_tensor

def batch(data):
    if data.ndim == 3:
        return data.unsqueeze(0)
    elif data.ndim == 2: # we assume gray scale here
        return data.unsqueeze(0).unsqueeze(0)
    return data

def normalize_noise(x):
    ''' normalize x when its range lays on [-1, 1] or [0, 1]'''
    return x.clamp(-1, 1)

def clip_to_noise(x):
    ''' maps [0, 1] to [-1, 1]'''
    return 2.0 * x - 1.0

def clip_to_img(x):
    ''' maps [-1, 1] to [0, 1]'''
    return (x + 1.0) / 2.0

def clip_to_pixel(x):
    ''' maps [-1, 1] to [0,255]'''
    return (255 * normalize(x))

def clip_to_noise_pixel(x):
    ''' maps [0, 255] to [-1, 1]'''
    return clip_to_noise(x / 255.0)

def img_range(x):
    x_ = x.detach()
    return (x_.min().item(), x_.max().item())

def make_mask(image_size, mask_type="box", mask_size=None, mask_density=None):
    bsz, c, h, w = image_size

    if mask_type == "box":
        h_, w_ = mask_size

        idx = random.randint(0, h - h_)
        idy = random.randint(0, w - w_)

        mask = torch.ones((bsz, c, h, w), dtype=torch.bool)
        mask[:,:,idx:idx+h_+1,idy:idy+w_+1] = False

        return mask

    elif mask_type == "random":
        rho = mask_density

        mask_float = torch.rand((h,w))
        mask = mask_float > rho
        mask = mask.unsqueeze(0)

        mask = torch.cat([mask] * c, dim=0)
        mask = mask.unsqueeze(0)

        return mask

    else:
        return ValueError(f"Unknown mask type {mask_type}")
    

# This blur class was taken from the original paper
class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k
