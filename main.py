import torch
from modules.backbones import build_backbone
from modules.headers import LinearHeader, ArcFaceHeader, MagFaceHeader
from face import FaceRecognitionModel
from data import LFWDataset

import albumentations as A

def main():
    embed_dim = 512

    preprocessing = A.Compose([
        A.CenterCrop(112, 112),
        A.Normalize(),
        A.HorizontalFlip()
    ])

    train_dataset = LFWDataset(mode='train', transform=preprocessing)
    test_dataset = LFWDataset(mode='test', transform=preprocessing)
    n_classes = train_dataset.get_n_identities()

    # build backbone, header and ce loss
    backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)
    header = ArcFaceHeader(in_features=embed_dim, out_features=n_classes)
    loss = torch.nn.CrossEntropyLoss()

    # create the face recognition model wrapper
    face_model = FaceRecognitionModel(backbone=backbone, header=header, loss=loss)

    # fit the face recognition model to the dataset
    face_model.fit(
        train_dataset=train_dataset,
        epochs=10,
        batch_size=32,
        lr=1e-2,
        device=torch.device('cuda')
        )

    embedding1, embedding2 = face_model.encode(imgs=[img1, img2])


if __name__ == '__main__':
    main()
