from .face_dataset import FaceDataset
from .util import http_get, extract_archive

import os
import numpy as np

class CelebADataset(FaceDataset):

    def __init__(self, download=True, name='celeba', aligned=True, split='test', **kwargs):
        assert aligned
        super(CelebADataset, self).__init__(name=name, **kwargs)

        self.download = download
        self.aligned = aligned
        self.split = split

        celeba_ids_file = os.path.join(self.root, self.name, "identity_CelebA.txt")
        celeba_partition_file = os.path.join(self.root, self.name, "list_eval_partition.txt")

        self.img_paths = [os.path.join(self.root, self.name, 'images', img_file) for img_file in os.listdir(os.path.join(self.root, self.name, 'images'))]
        self.img_id_labels = self.__read_celeba_ids_from_file(celeba_ids_file)

        if split != 'all':
            split_idxs = self.__read_celeba_split_mask_from_file(celeba_partition_file, split=split)
            self.reduce_to_sample_idxs(split_idxs)


    def download_dataset(self):
        raise NotImplementedError


    def __read_celeba_ids_from_file(self, ids_path):
        id_labels = []
        with open(ids_path, 'r') as f:
            for line in f.readlines()[1:]:
                label = int(line.strip().split()[1]) - 1
                id_labels.append(label)
        return id_labels


    def __read_celeba_split_mask_from_file(self, partition_path, split):
        split_label = {'train': 0, 'val': 1, 'test': 2}[split]

        partition_labels = []
        with open(partition_path, 'r') as f:
            for line in f.readlines()[1:]:
                partition = int(line.strip().split()[1])
                partition_labels.append(partition)

        return np.where(np.array(partition_labels) == split_label)[0]
