import torch
import torchvision

from femb.backbones import build_backbone
from femb.headers import LinearHeader, SphereFaceHeader, CosFaceHeader, ArcFaceHeader, MagFaceHeader
from femb.evaluation import VerificationEvaluator
from femb.data import LFWDataset, CelebADataset
from femb import FaceEmbeddingModel

import albumentations as A


def main():
    embed_dim = 128

    transform = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        A.HorizontalFlip(),
    ])

    train_dataset = LFWDataset(split='train', aligned=True, albu_transform=transform)
    val_dataset = LFWDataset(split='test', aligned=True, albu_transform=transform)

    train_n_classes = train_dataset.get_n_identities()
    val_n_classes = val_dataset.get_n_identities()

    print(f"Train Dataset - #images: {len(train_dataset)} - #ids: {train_n_classes}")
    print(f"Val Dataset - #images: {len(val_dataset)} - #ids: {val_n_classes}")

    # build backbone, header and ce loss
    backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)

    # header = LinearHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = SphereFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = CosFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    header = ArcFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = MagFaceHeader(in_features=embed_dim, out_features=train_n_classes)

    loss = torch.nn.CrossEntropyLoss()

    # create the face recognition model wrapper
    face_model = FaceEmbeddingModel(backbone=backbone, header=header, loss=loss)

    evaluator = VerificationEvaluator(similarity='cos')
    optimizer = torch.optim.Adam(params=face_model.params, lr=1e-1)

    lr_global_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[8000, 10000, 160000], gamma=0.1)

    # fit the face embedding model to the dataset
    face_model.fit(
        train_dataset=train_dataset,
        batch_size=32,
        device='cuda',
        optimizer=optimizer,
        lr_epoch_scheduler=None,
        lr_global_step_scheduler=None,
        evaluator=evaluator,
        val_dataset=val_dataset,
        evaluation_steps=10,
        max_training_steps=20000,
        max_epochs=0,
        tensorboard=True
        )


if __name__ == '__main__':
    main()
