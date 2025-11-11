import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..evaluation.metrics import Metric
from ..loggers.logger import Logger
from ..model.unet import UNet
from ..utils import prepare_image

Tensor = torch.Tensor
Device = torch.device

def interpolate(v_start: float, v_end: float, time_steps: int, mode: str = "linear") -> Tensor:
    # TODO : implement other interpolation method (cosine)
    if mode == "linear":
        out = np.linspace(v_start, v_end, time_steps)
        return torch.from_numpy(out).float()
    else:
        raise ValueError(f"Unknown mode for interpolation mode={mode}")

class Diffusion():

    # in this setting we suppose T = N
    def __init__(self,
                size_noise: tuple,
                beta_start: float,
                beta_end: float,
                time_steps: int,
                # model: UNet,
                device: Device,
                logger: Logger,
                metric: Metric,
                measurement_step: bool = None,
                enable_batch_grad: bool = False,
                enable_log: bool = True,
                save_dir: str = './results/'):
        self.size_noise = size_noise

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        self.scaling_term = 1000.0 / time_steps
        self.time_tensor = self.scaling_term * torch.arange(time_steps, dtype=torch.float, device=device).unsqueeze(-1).repeat_interleave(size_noise[0], dim=-1)

        # self.model = model
        self.device = device

        self.measurement_step = measurement_step

        self.logger = logger
        self.metric = metric
        self.enable_batch_grad = enable_batch_grad
        self.enable_log = enable_log

        self.save_dir = save_dir

        # self.model.eval()
        self.get_beta()
        self.get_alpha()
        self.get_sigma()

    def get_beta(self):
        self.betas = interpolate(self.beta_start, self.beta_end, self.time_steps)

    def get_alpha(self):
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev

        self.coef_approx_1 = torch.sqrt(1.0 / alphas_bar)
        self.coef_approx_2 = - torch.sqrt(1.0 / alphas_bar - 1)
        self.coef_mean_1 = self.betas * torch.sqrt(alphas_bar_prev) / (1.0-alphas_bar)
        self.coef_mean_2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)

    def get_sigma(self):
        # fixed sigma scheduler
        self.sigma = torch.sqrt(self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar))
        self.sigma[0] = 0.0

    def to(self, device: Device):
        self.model.to(device)
        self.betas.to(device)
        self.alphas.to(device)
        self.alphas.to(device)

    def compute_mean(self, x_i: Tensor, eps: Tensor, step: int) -> Tensor:
        # line 4 of algorithm 1
        approx_x = self.coef_approx_1[step] * x_i + \
                   self.coef_approx_2[step] * eps

        # line 6 of algorithm 1
        posterior_mean = self.coef_mean_1[step] * approx_x + \
                         self.coef_mean_2[step] * x_i

        return approx_x, posterior_mean

    def compute_variance(self, s_theta_var: Tensor, step: int) -> Tensor:
        # Implementation of fixed sigma
        # return self.sigma[step]


        # Implementation of variational variance using unet learned variance prediction model
        if step == 0:
            return torch.zeros_like(s_theta_var)

        noise_rate = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        min_var = torch.log(torch.cat([torch.tensor([noise_rate[1].item()]), noise_rate[1:]]))[step]
        max_var = torch.log(self.betas)[step]

        weight = (s_theta_var + 1.) / 2.

        gamma = weight * max_var + (1. - weight) * min_var

        return torch.exp(gamma / 2.)

    def one_step_sampling(self, x: Tensor, model: UNet,  step: int) -> tuple:
        time = self.time_tensor[step]

        if self.enable_batch_grad:
            x = x.requires_grad_()
            output = model(x, time)
        else:
            with torch.no_grad():
                output = self.model(x, time)
        s_hat, s_hat_variance = torch.split(output, x.size(1), dim=1)

        # with torch.no_grad():
        #     s_hat, s_hat_variance = torch.split(model(x, time), x.size(1), dim=1)

        noise = torch.randn_like(x)
        approx_x, posterior_mean = self.compute_mean(x, s_hat, step)
        sigma_noise = self.compute_variance(s_hat_variance, step)

        x_next = posterior_mean + sigma_noise * noise

        return x_next, approx_x

    def full_sampling(self, model: UNet, y: Tensor = None, ground_truth: Tensor = None) -> Tensor:
        x_prev = torch.randn(self.size_noise, device=self.device)

        step_set = tqdm(range(self.time_steps)[::-1], position=1) #if self.enable_log else range(self.time_steps)[::-1]
        for step in step_set:
            x_prev_unconditional, approx_x = self.one_step_sampling(x_prev, model, step)

            if self.measurement_step:
                x_prev, norm = self.measurement_step(x_next=x_prev_unconditional, x_prev=x_prev, approx_x=approx_x, y=y, step=step)
                x_prev = x_prev.detach()
                step_set.set_postfix({'norm': norm.item()}, refresh=False)
            else:
                x_prev = x_prev_unconditional.detach()          


            if ground_truth is not None and self.enable_log:
                self.logger.update_stats(**self.metric.eval(x_prev, ground_truth))

            if step % 200 == 0:
                file_path = os.path.join(self.save_dir, f"progress/x_{str(step).zfill(4)}.png")
                plt.imsave(file_path, prepare_image(x_prev))
                plt.close('all')
        
        self.logger.update_stats(**self.metric.eval(x_prev, ground_truth))
        return x_prev