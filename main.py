import os
import argparse
import yaml
from functools import partial

import torch
import matplotlib.pyplot as plt
from deepinv.utils.plotting import plot


from dps.model.unet import UNet
from dps.diffusion.diffusion import Diffusion
from dps.diffusion.measurement import get_operator, get_noise
from dps.utils import prepare_image, batch, load_image, normalize
from dps.loader.data_loader import prepare_dataset, prepare_dataloader
from dps.loader.config_loader import load_parser, get_sample_size
from dps.evaluation.metrics import get_metric
from dps.loggers.logger import Logger

parser = load_parser()

out_dir = parser["out_dir"]
seed = parser["seed"]
batch_size = parser["batch_size"]
channels = parser["channels"]
T = parser["steps"]
sampling_size = parser["sampling_size"]
device_str = parser["device"]
device = torch.device(device_str)
sample_size = get_sample_size(parser)

conditional_cfg = parser["conditional"]
use_conditional = conditional_cfg["use_conditional"]
dataset_name = conditional_cfg["dataset"]
dataset_dir = conditional_cfg["dataset_dir"]
operator = conditional_cfg["operator"]
noise = conditional_cfg["noise"]

torch.manual_seed(seed)

beta_start = 1e-4
beta_end = 2e-2

# Choose operator and noise
operator_cfg = conditional_cfg["operator"]
operator_cfg["sample_size"] = sample_size
operator = get_operator(device=device, **operator_cfg)

noise_cfg = conditional_cfg["noise"]
noise = get_noise(**noise_cfg)
scale = conditional_cfg["scale"]

# Choose metric
metric_cfg = parser["metric"]
metric = get_metric(**metric_cfg)

# Prepare logger
logger = Logger(metric.get_names())


# put model choice inside sampler
ckpt_path = parser["ckpt_file"]
model = UNet(channels,channels)
model.load_ckpt(ckpt_path)
model.to(device)
model.eval()

dataset = prepare_dataset(dataset_name=dataset_name, dataset_dir=dataset_dir)
dataloader = prepare_dataloader(dataset, batch_size=batch_size)

if use_conditional:
    measurement_step = partial(operator.grad, noise=noise, scale=scale)
else:
    measurement_step = None

sampler = Diffusion(size_noise=sample_size, beta_start=beta_start, beta_end=beta_end,
                    time_steps=T, model=model, device=device, logger=logger, metric=metric, measurement_step=measurement_step,
                    enable_log=True, enable_batch_grad=use_conditional)

for index, img in enumerate(dataloader):
    if isinstance(img, list):
        img = img[0]

    print(img.shape)
    img = img.to(device)
    batch_name = str(index).zfill(4) + ".png"

    # check batch size reminder !!!!!!!!!!!!!!!!!
    # lazy fix for now
    if img.size(0) < batch_size:
        break

    # perform noisy transformation
    if use_conditional:
        y = operator.transform(img)
        y = noise.apply(y).to(device)
    else:
        y = None


    output = sampler.full_sampling(y=y, ground_truth=img)

    logger.write_step(-1)

    if use_conditional:
        plt.imsave(os.path.join(out_dir, "truth" + batch_name), prepare_image(img))
        plt.imsave(os.path.join(out_dir, "rec" + batch_name), prepare_image(output))
        plt.imsave(os.path.join(out_dir, "measure" + batch_name), prepare_image(y))
    else:
        plt.imsave(os.path.join(out_dir, batch_name), prepare_image(output))

del model, dataset, dataloader