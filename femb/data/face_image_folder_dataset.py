from .face_dataset import FaceDataset

import os
import logging


class FaceImageFolderDataset(FaceDataset):

    def __init__(self, auto_initialize=True, **kwargs):
        super(FaceImageFolderDataset, self).__init__(**kwargs)

        self.img_paths = []
        self.img_ids = []
        self.img_id_labels = []

        if auto_initialize:
            self.init_from_directories()


    def init_from_directories(self):
        if not self.dataset_exists():
            logging.warning(f"The dataset {self.name} does not contain any images under {os.path.join(self.root, self.name, 'images')}")
            return

        logging.info(f"Creating a FaceImageFolderDataset ({self.name}) with data from {os.path.join(self.root, self.name, 'images')}.")

        images_dir = os.path.join(self.root, self.name, 'images')

        for label, identity in enumerate(os.listdir(images_dir)):
            id_path = os.path.join(images_dir, identity)

            for img_file in os.listdir(id_path):
                self.img_paths.append(os.path.join(id_path, img_file))
                self.img_ids.append(identity)
                self.img_id_labels.append(label)


    def dataset_exists(self):
        images_dir = os.path.join(self.root, self.name, 'images')
        return os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0


    def __getitem__(self, idx):
        sample = super(FaceImageFolderDataset, self).__getitem__(idx)
        sample['id'] = self.img_ids[idx]
        return sample
