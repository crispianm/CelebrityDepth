# dataset

from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceDepthDataset(Dataset):
    def __init__(self, celeba_folder, depth_folder, transform=None):
        self.celeba_folder = celeba_folder
        self.depth_folder = depth_folder
        self.transform = transform
        self.celebrity_filenames = os.listdir(celeba_folder)
        self.depth_filenames = os.listdir(depth_folder)

    def __len__(self):
        return len(self.celebrity_filenames)

    def __getitem__(self, index):
        celeba_path = os.path.join(self.celeba_folder, self.celebrity_filenames[index])
        depth_path = os.path.join(self.depth_folder, self.depth_filenames[index])

        face = Image.open(celeba_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        # depth = depth.expand(3,*depth.shape[1:])

        if self.transform is not None:
            face = self.transform(face)
            depth = self.transform(depth)

        return depth, face
