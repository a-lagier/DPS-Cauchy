import torch
from torchvision import transforms
import torchvision.datasets
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

def prepare_dataset(dataset_name, dataset_dir="./datasets/"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])

    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    else:
        dataset = SamplesDataset(os.path.join(dataset_dir, dataset_name), transform=transform)
        return dataset
    
    return dataset

def prepare_dataloader(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

