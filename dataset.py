# dataset

from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceDepthDataset(Dataset):
    def __init__(self, image_folder, depth_folder, transform=None):
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.transform = transform
        self.image_filenames = os.listdir(image_folder)
        self.depth_filenames = os.listdir(depth_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_filenames[index])
        depth_path = os.path.join(self.depth_folder, self.depth_filenames[index])

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        # depth = depth.expand(3,*depth.shape[1:])

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, image


# if __name__ == "__main__":
#     dataset = FaceDepthDataset("./celeba/a_test", "./celeba/d_test")
#     loader = DataLoader(dataset, batch_size=5)
#     for x, y in loader:
#         print(x.shape)
#         import sys

#         sys.exit()
