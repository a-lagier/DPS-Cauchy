import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils import normalize, prepare_image, clip_to_img, clip_to_noise, img_range, normalize_noise, clip_to_pixel, clip_to_noise_pixel


def interpolate(v_start, v_end, time_steps, mode="linear"):
    # TODO : implement other interpolation method (cosine)
    if mode == "linear":
        out = np.linspace(v_start, v_end, time_steps)
        return torch.from_numpy(out).float()
    else:
        raise ValueError(f"Unknown mode for interpolation mode={mode}")

class Diffusion():

    # in this setting we suppose T = N
    def __init__(self,
                size_noise,
                beta_start,
                beta_end,
                time_steps,
                model,
                device,
                logger,
                metric,
                measurement_step=None,
                enable_batch_grad=False,
                enable_log=True):
        self.size_noise = size_noise

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = time_steps
        self.scaling_term = 1000.0 / time_steps

        self.model = model
        self.device = device

        self.measurement_step = measurement_step


        self.logger = logger
        self.metric = metric
        self.enable_batch_grad = enable_batch_grad
        self.enable_log = enable_log

        self.model.eval()
        self.get_beta()
        self.get_alpha()

    def get_beta(self):
        self.betas = interpolate(self.beta_start, self.beta_end, self.time_steps)

    def get_alpha(self):
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev

        self.coef_mean_1 = torch.sqrt(1.0 / alphas_bar)
        self.coef_mean_2 = - torch.sqrt(1.0 / alphas_bar - 1)
        self.coef_next_1 = self.betas * torch.sqrt(alphas_bar_prev) / (1.0-alphas_bar)
        self.coef_next_2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)

    def to(self, device):
        self.model.to(device)
        self.betas.to(device)
        self.alphas.to(device)
        self.alphas.to(device)

    def compute_mean(self, x_i, s_theta_mean, step):
        # line 4 of algorithm 1
        approx_x = self.coef_mean_1[step] * x_i + \
                   self.coef_mean_2[step] * s_theta_mean
        # line 6 of algorithm 1
        posterior_mean = self.coef_next_1[step] * approx_x + \
                         self.coef_next_2[step] * x_i

        return approx_x, posterior_mean

    def compute_variance(self, s_theta_var, step):
        if step == 0:
            return torch.zeros_like(s_theta_var)

        return torch.sqrt(self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar))[step]

        # Implementation of variational variance using unet variance prediction model
        # Unused for now (experimentally a fixed sigma_t is better)

        # noise_rate = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        # min_var = torch.log(torch.cat([torch.tensor([noise_rate[1].item()]), noise_rate[1:]]))[step]
        # max_var = torch.log(self.betas)[step]

        # weight = (s_theta_var + 1.) / 2.

        # gamma = weight * max_var + (1. - weight) * min_var

        # return torch.exp(gamma / 2.)

    def get_sigma_denoiser(self, step):
        alpha_bar = self.alphas_bar[step]

        return torch.sqrt((1. - alpha_bar) / alpha_bar)

    def one_step_sampling(self, x, step):
        assert step >= 0

        time = torch.tensor([step * self.scaling_term] * x.size(0), device=self.device).float()
        
        if self.enable_batch_grad:
            x = x.requires_grad_()
            s_hat, s_hat_variance = torch.split(self.model(x, time), x.size(1), dim=1)
        else:
            s_hat, s_hat_variance = torch.split(self.model(x, time).detach(), x.size(1), dim=1)

        noise = torch.randn_like(x)

        approx_x, posterior_mean = self.compute_mean(x, s_hat, step)
        sigma_noise = self.compute_variance(s_hat_variance, step)

        x_next = posterior_mean + sigma_noise * noise

        return x_next, approx_x

    def full_sampling(self, y=None, ground_truth=None):
        x_prev = torch.randn(self.size_noise, device=self.device)

        step_set = tqdm(range(self.time_steps)[::-1]) if self.enable_log else range(self.time_steps)[::-1]

        for step in step_set:
            x_prev_unconditional, approx_x = self.one_step_sampling(x_prev, step)

            if self.measurement_step:
                x_prev = self.measurement_step(x_next=x_prev_unconditional, x_prev=x_prev, approx_x=approx_x, y=y)

            if ground_truth is not None:
                self.logger.update_stats(**self.metric.eval(x_prev, ground_truth))

            if step % 100 == 0:
                file_path = os.path.join("./generated_data/", f"progress/x_{str(step).zfill(4)}.png")
                plt.imsave(file_path, prepare_image(x_prev))
        return x_prev


# old code
# class ConditionalDiffusion(Diffusion):

#     def __init__(self, size_noise, beta_start, beta_end, time_steps, model, device, transform, noise, enable_log=True):
#         super().__init__(size_noise, beta_start, beta_end, time_steps, model, device, enable_batch_grad=True)

#         self.transform = transform
#         self.noise_distribution = noise

#         self.enable_log = enable_log

#     def grad_transform(self, x, approx_x, target):
#         transformed_approx_x = self.transform.transform(approx_x)
#         log_likelihood = self.noise_distribution.log_likelihood(target, transformed_approx_x)
#         grad = torch.autograd.grad(outputs=log_likelihood, inputs=x)[0]

#         return grad.detach()


#     def one_step_conditional_denoising(self, x, target, step):
#         x_prime, approx_x = self.one_step_denoising(x, step)
        
#         zeta = 0.01
#         grad_ = self.grad_transform(x, approx_x, target)

#         x_prev = x_prime - zeta * grad_
#         return x_prev

#     def full_conditional_denoising(self, target):
#         x_prev = torch.randn(self.size_noise)

#         step_set = tqdm(range(self.time_steps)[::-1]) if self.enable_log else range(self.time_steps)[::-1]

#         for step in step_set:
#             x_prev = self.one_step_conditional_denoising(x_prev, target, step)
#             if step % 100 == 0:
#                 file_path = os.path.join("./generated_data/", f"progress/x_{str(step).zfill(4)}.png")
#                 plt.imsave(file_path, prepare_image(x_prev))
#         return x_prev
