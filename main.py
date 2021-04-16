import torch
import torchvision

from femb.backbones import build_backbone
from femb.headers import LinearHeader, SphereFaceHeader, CosFaceHeader, ArcFaceHeader, MagFaceHeader
from femb.evaluation import VerificationEvaluator
from femb.data import LFWDataset, CelebADataset
from femb import FaceEmbeddingModel

def main():
    # specify the size of the embeddings
    embed_dim = 256

    # preprocessing transform (assuming alignment and so on)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((112, 112)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        torchvision.transforms.RandomHorizontalFlip()
        ])

    # loading the face dataset
    train_dataset = LFWDataset(split='train', aligned=True, transform=transform)
    val_dataset = LFWDataset(split='test', aligned=True, transform=transform)

    # shrink the datasets due to limited compute capabilities
    #train_dataset.reduce_to_N_identities(1000)
    #val_dataset.reduce_to_N_identities(100)

    train_n_classes = train_dataset.get_n_identities()
    val_n_classes = val_dataset.get_n_identities()

    print(f"Train Dataset - #images: {len(train_dataset)} - #ids: {train_n_classes}")
    print(f"Val Dataset - #images: {len(val_dataset)} - #ids: {val_n_classes}")

    # build the backbone embedding network
    backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)

    # create one of the face recognition headers
    #header = LinearHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = SphereFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = CosFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    # header = ArcFaceHeader(in_features=embed_dim, out_features=train_n_classes)
    header = MagFaceHeader(in_features=embed_dim, out_features=train_n_classes)

    # create the ce loss
    loss = torch.nn.CrossEntropyLoss()

    # create the face recognition model wrapper
    face_model = FaceEmbeddingModel(backbone=backbone, header=header, loss=loss)

    # create the verification evaluator
    evaluator = VerificationEvaluator(similarity='cos')

    # specify the optimizer (and a scheduler)
    optimizer = torch.optim.SGD(params=face_model.params, lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2000, 4000, 6000], gamma=0.1)

    # fit the face embedding model to the dataset
    face_model.fit(
        train_dataset=train_dataset,
        batch_size=32,
        device='cuda',
        optimizer=optimizer,
        lr_epoch_scheduler=None,
        lr_global_step_scheduler=scheduler,
        evaluator=evaluator,
        val_dataset=val_dataset,
        evaluation_steps=1000,
        max_training_steps=10000,
        max_epochs=3,
        tensorboard=True
        )


if __name__ == '__main__':
    main()
