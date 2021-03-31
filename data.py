import os
import torch
import torchvision
import cv2

from util import http_get, extract_archive


class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, mode='test', aligned=True, dataset_dir='datasets/', transform=None):

        self.transform = transform

        lfw_path = os.path.join(dataset_dir, 'lfw-dataset')

        if aligned:
            subfolder = 'lfw-deepfunneled'
            download_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
        else:
            subfolder = 'lfw'
            download_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"

        tgz_path = os.path.join(dataset_dir, os.path.basename(download_url))

        # download lfw .tgz file if necessary
        if not os.path.isfile(tgz_path):
            http_get(url=download_url, path=tgz_path)
        # extract it if necessary
        if not os.path.isdir(os.path.join(lfw_path, subfolder)):
            extract_archive(archive=tgz_path, destination=lfw_path)

        # download the correct pairs/people split files for the specified view if necessary and read them
        if mode == 'train':
            people_path = os.path.join(lfw_path, 'peopleDevTrain.txt')

            if not os.path.isfile(people_path):
                http_get(url="http://vis-www.cs.umass.edu/lfw/peopleDevTrain.txt", path=people_path)

            #self.pairs = self.__read_lfw_pairs_from_file(pairs_path)
            self.people = self.__read_lfw_people_from_file(people_path)

        elif mode == 'test':
            people_path = os.path.join(lfw_path, 'peopleDevTest.txt')

            if not os.path.isfile(people_path):
                http_get(url="http://vis-www.cs.umass.edu/lfw/peopleDevTest.txt", path=people_path)

            self.people = self.__read_lfw_people_from_file(people_path)

        elif mode == 'benchmark':
            raise NotImplementedError

        self.img_paths, self.img_identities, self.img_labels = self.__get_lfw_img_paths(root=os.path.join(lfw_path, subfolder))


    def __read_lfw_people_from_file(self, people_path):
        peoples = []
        with open(people_path, 'r') as f:
            for line in f.readlines()[1:]:
                identity = line.strip().split()[0]
                peoples.append(identity)
        return peoples


    def __get_lfw_img_paths(self, root):
        assert self.people is not None
        assert os.path.isdir(root)

        img_paths = []
        img_identities = []
        img_labels = []
        for label, identity in enumerate(self.people):
            id_path = os.path.join(root, identity)
            img_paths.extend([os.path.join(id_path, img_file) for img_file in os.listdir(id_path)])
            img_identities.extend([identity] * len(os.listdir(id_path)))
            img_labels.extend([label] * len(os.listdir(id_path)))
        return img_paths, img_identities, img_labels


    def __len__(self):
        return len(self.img_paths)


    def get_n_identities(self):
        return len(self.people)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self):
            raise IndexError

        img_path = self.img_paths[idx]
        img_identity = self.img_identities[idx]
        img_label = self.img_labels[idx]

        img = cv2.imread(img_path)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = torchvision.transforms.functional.to_tensor(img).float()

        return {'img': img, 'identity': img_identity, 'label': img_label}



if __name__ == '__main__':

    dataset = LFWDataset()
