from .face_image_folder_dataset import FaceImageFolderDataset
from .util import http_get, extract_archive

import os
import numpy as np

class LFWDataset(FaceImageFolderDataset):

    def __init__(self, download=True, aligned=True, split='test', **kwargs):
        super(LFWDataset, self).__init__(name='lfw' if not aligned else 'lfw-deepfunneled', auto_initialize=False, **kwargs)

        self.download = download
        self.aligned = aligned
        self.split = split

        if download and not self.dataset_exists():
            self.download_dataset()

        self.init_from_directories()

        people_train_path = os.path.join(self.root, self.name, 'peopleDevTrain.txt')
        people_test_path = os.path.join(self.root, self.name, 'peopleDevTest.txt')

        people = []
        if self.split in ['train', 'all']:
            people.extend(self.__read_lfw_people_from_file(people_train_path))
        if self.split in ['test', 'all']:
            people.extend(self.__read_lfw_people_from_file(people_test_path))

        if split != 'all':
            split_idxs = np.where(np.isin(self.img_ids, people))[0]
            self.img_paths = [self.img_paths[idx] for idx in split_idxs]
            self.img_ids = [self.img_ids[idx] for idx in split_idxs]
            self.img_id_labels = [self.img_id_labels[idx] for idx in split_idxs]


    def download_dataset(self):
        if self.aligned:
            download_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
        else:
            download_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"

        tgz_path = os.path.join(self.root, os.path.basename(download_url))

        # download lfw .tgz file if necessary
        if not os.path.isfile(tgz_path):
            http_get(url=download_url, path=tgz_path)

        # extract it if necessary
        if not os.path.isdir(os.path.join(self.root, self.name)):
            extract_archive(archive=tgz_path, destination=os.path.join(self.root, self.name))
            os.rename(os.path.join(self.root, self.name, 'lfw' if not self.aligned else 'lfw-deepfunneled'), (os.path.join(self.root, self.name, 'images')))

        people_train_path = os.path.join(self.root, self.name, 'peopleDevTrain.txt')
        people_test_path = os.path.join(self.root, self.name, 'peopleDevTest.txt')

        self.people = []
        if self.split in ['train', 'all'] and not os.path.isfile(people_train_path):
            http_get(url="http://vis-www.cs.umass.edu/lfw/peopleDevTrain.txt", path=people_train_path)
        if self.split in ['test', 'all'] and not os.path.isfile(people_test_path):
            http_get(url="http://vis-www.cs.umass.edu/lfw/peopleDevTest.txt", path=people_test_path)


    def __read_lfw_people_from_file(self, people_path):
        peoples = []
        with open(people_path, 'r') as f:
            for line in f.readlines()[1:]:
                identity = line.strip().split()[0]
                peoples.append(identity)
        return peoples
