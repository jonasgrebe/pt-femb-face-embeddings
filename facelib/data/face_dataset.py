import torch

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, name=None, root='datasets/', transform=None):
        assert name
        self.name = name
        self.root = root
        self.transform = transform

    def __len__(self):
        return self.get_n_images()

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_n_identities(self):
        raise NotImplementedError

    def get_n_images(self):
        raise NotImplementedError

    def split(self, level='ids'):
        raise NotImplementedError
