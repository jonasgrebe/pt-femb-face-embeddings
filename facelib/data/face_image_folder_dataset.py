from .face_dataset import FaceDataset

import os
import torch
import numpy as np

from imageio import imread
import torchvision

class FaceImageFolderDataset(FaceDataset):

    def __init__(self, **kwargs):
        super(FaceImageFolderDataset, self).__init__(**kwargs)

        self.img_paths = []
        self.img_ids = []
        self.img_labels = []

        if self.dataset_exists():
            self.init_from_directories()
        else:
            print(f"WARNING: The dataset {self.name} does not contain any images under {os.path.join(self.root, self.name, 'images')}")


    def init_from_directories(self):
        if not self.dataset_exists():
            return

        images_dir = os.path.join(self.root, self.name, 'images')

        for label, identity in enumerate(os.listdir(images_dir)):
            id_path = os.path.join(images_dir, identity)

            for img_file in os.listdir(id_path):
                self.img_paths.append(os.path.join(id_path, img_file))
                self.img_ids.append(identity)
                self.img_labels.append(label)


    def dataset_exists(self):
        images_dir = os.path.join(self.root, self.name, 'images')
        return os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self):
            raise IndexError

        img_path = self.img_paths[idx]
        img_id = self.img_ids[idx]
        img_label = self.img_labels[idx]

        img = imread(img_path)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torchvision.transforms.functional.to_tensor(img).float()

        return {'image': img, 'identity': img_id, 'label': img_label}


    def get_n_images(self):
        return len(self.img_paths)


    def get_n_identities(self):
        return np.max(self.img_labels) + 1
