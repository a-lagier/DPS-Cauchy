import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt


class SamplesDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(image_dir)]
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = plt.imread(image_path)[..., :3]
        
        if self.transform:
            image = self.transform(image)
        return image

def prepare_dataset(dataset_name: str, dataset_dir: str = "./datasets/") -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])

    if dataset_name == "CIFAR10":
        dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    elif dataset_name == "CelebA":
        # The link for downloading CelebA is temporary down
        # dataset = CelebA(root=dataset_dir, transform=transform, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(178),
            transforms.Resize(256),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = SamplesDataset(os.path.join(dataset_dir, 'celeba'), transform=transform)
    elif dataset_name == "MiniImageNet":
        pass
        # dataset = torchvision.datasets.MiniImageNet(root=dataset_dir, transform=transform, download=True)
    else:
        dataset = SamplesDataset(os.path.join(dataset_dir, dataset_name), transform=transform)
        return dataset
    
    return dataset

def prepare_dataloader(dataset: Dataset, batch_size: str = 1) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

