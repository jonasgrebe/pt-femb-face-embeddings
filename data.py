import os
import torch
from util import http_get, extract_archive


class FaceDataset(torch.utils.data.Dataset):

    def __init__(name='lfw'):
        pass




class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, mode='test', aligned=True, dataset_dir='datasets/'):

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

        self.img_paths, self.img_identities = self.__get_lfw_img_paths(root=os.path.join(lfw_path, subfolder))


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
        for label, identity in enumerate(self.people):
            id_path = os.path.join(root, identity)
            img_paths.extend([os.path.join(identity, img_file) for img_file in os.listdir(id_path)])
            img_identities.extend([label] * len(os.listdir(id_path)))
        return img_paths, img_identities


    def __get_item__(self, idx):

        img_path = self.img_paths[idx]
        img_identity = self.img_identities[idx]

        img = cv2.read(img_path)

        return img, img_identity



if __name__ == '__main__':

    dataset = LFWDataset()
