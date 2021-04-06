import torch

from imageio import imread
import torchvision
import numpy as np

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, name=None, root='datasets/', transform=None):
        assert name
        self.name = name
        self.root = root
        self.transform = transform

        self.img_paths = None
        self.img_id_labels = None


    def __len__(self):
        return self.get_n_images()


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        img_path = self.img_paths[idx]
        img_label = self.img_id_labels[idx]

        img = imread(img_path)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torchvision.transforms.functional.to_tensor(img).float()

        return {'image': img, 'label': img_label, 'idx': idx}


    def get_n_identities(self):
        return np.max(self.img_id_labels) + 1


    def get_n_images(self):
        return len(self.img_paths)
