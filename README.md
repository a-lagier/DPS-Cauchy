# DPS with non exponential noise

## Getting Started

Clone the repository

```
git clone https://github.com/a-lagier/DPS-reimplementation

cd diffusion-posterior-sampling
```

### Other code projects to import
We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

### Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch torchvision deepinv lpips gdown datetime
```

### Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

## Inference
The dir configs/ will contain basic DPS configs ready to be ran with the following command
```
python main.py --cfg configs/motion-blur-cauchy-celeba.yaml
```
Here is the content of the file configs/motion-blur-cauchy-celeba.yaml
```
out_dir: ./results
device: cuda:0
sampling_size: 256 # if conditional choose it accordingly with dataset image size
model: unet
ckpt_file: ./models/ffhq_10m.pt
steps: 1000
seed: 0
channels: 3
batch_size: 10
conditional:
  use_conditional: True
  scale: 1.0
  dataset:
    name: CelebA
    dataset_dir: ./datasets/
    num_images: 250
  operator:
    name: motion-blur
    kernel_size : 61
    intensity: 0.5
  noise:
    name: cauchy
    scale: 1.0
metric:
  name: [lpips, psnr]
```

## TODO : code implementation

* U-net (ResNet, Attention block,...) : adapt each method with time integration
* Gaussian Diffusion
* Conditional diffusion
* Transformation y = Ax + n

## Edit packages

* "/Users/alexandre/opt/anaconda3/envs/DPS/lib/python3.8/site-packages/deepinv/training/trainer.py", line 138, in Trainer
  scheduler: torch.optim.lr_scheduler.LRScheduler = None change to _LRScheduler
