import torch
import torchvision

from femb.backbones import build_backbone
from femb.headers import LinearHeader, ArcFaceHeader, MagFaceHeader, SphereFaceHeader
from femb.evaluation import VerificationEvaluator
from femb.data import LFWDataset, CelebADataset
from femb import FaceEmbeddingModel
import albumentations as A

import logging

def main():
    logging.basicConfig(filename='example.log', level=logging.INFO)

    embed_dim = 256

    preprocessing = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=(0.5,0.5,0.5), std=(1,1,1)),
        A.HorizontalFlip(),
    ])

    train_dataset = CelebADataset(split='train', transform=preprocessing)
    val_dataset = CelebADataset(split='val', transform=preprocessing)

    #train_dataset = eval_dataset
    n_classes = train_dataset.get_n_identities()

    # build backbone, header and ce loss
    backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)
    header = MagFaceHeader(in_features=embed_dim, out_features=n_classes)
    loss = torch.nn.CrossEntropyLoss()

    # create the face recognition model wrapper
    face_model = FaceEmbeddingModel(backbone=backbone, header=header, loss=loss)

    # header = TripletHeader(online=True, mining='semi-hard')
    # loss = RankingLoss()
    # face_model = FaceEmbeddingModel(backbone='iresnet18', header=header, loss=loss)


    evaluator = VerificationEvaluator(val_dataset, similarity='cos')
    # evaluator = LFWVerificationEvaluator(similarity='cos')

    optimizer = torch.optim.SGD(params=face_model.params, lr=1e-1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 18, 22], gamma=0.1)

    # fit the face embedding model to the dataset
    face_model.fit(
        train_dataset=train_dataset,
        epochs=25,
        batch_size=32,
        device='cuda',
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        evaluator=evaluator,
        #output_path='output/'
        )

    img_path = "E:/GitHub/pt-arcface-magface/datasets/lfw-dataset/lfw-deepfunneled/Jennifer_Thompson/Jennifer_Thompson_0002.jpg"
    genuine_path = "E:/GitHub/pt-arcface-magface/datasets/lfw-dataset/lfw-deepfunneled/Jennifer_Thompson/Jennifer_Thompson_0001.jpg"
    imposter_path = "E:/GitHub/pt-arcface-magface/datasets/lfw-dataset/lfw-deepfunneled/Joe_Garner/Joe_Garner_0001.jpg"

    from imageio import imread

    imgs = [preprocessing(image=imread(path))['image'] for path in [img_path, genuine_path, imposter_path]]

    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(np.concatenate(imgs, axis=1))
    plt.show()

    embeddings = face_model.encode(imgs, device=device)

    from sklearn.metrics.pairwise import cosine_similarity

    print(cosine_similarity(embeddings))


if __name__ == '__main__':
    main()
