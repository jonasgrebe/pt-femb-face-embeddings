import torch

from modules.backbones import build_backbone
from modules.headers import LinearHeader, ArcFaceHeader, MagFaceHeader
from evaluation.biometric_evaluator import BiometricEvaluator
from data.lfw import LFWDataset
from face import FaceRecognitionModel

import albumentations as A

def main():
    embed_dim = 512

    preprocessing = A.Compose([
        A.CenterCrop(112, 112),
        A.Normalize(mean=(0.5,0.5,0.5), std=(1,1,1)),
        A.HorizontalFlip()
    ])

    train_dataset = LFWDataset(mode='train', aligned=True, transform=preprocessing)
    val_dataset = LFWDataset(mode='test', aligned=True, transform=preprocessing)


    #train_dataset = eval_dataset
    n_classes = train_dataset.get_n_identities()

    # build backbone, header and ce loss
    backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)
    header = ArcFaceHeader(in_features=embed_dim, out_features=n_classes)
    loss = torch.nn.CrossEntropyLoss()

    # create the face recognition model wrapper
    face_model = FaceRecognitionModel(backbone=backbone, header=header, loss=loss)

    evaluator = BiometricEvaluator(val_dataset, similarity='cos')

    # evaluator = LFWVerificationEvaluator(similarity='cos')

    device = torch.device('cuda')

    # fit the face recognition model to the dataset
    face_model.fit(
        train_dataset=train_dataset,
        epochs=25,
        batch_size=32,
        initial_lr=1e-1,
        device=device,
        evaluator=evaluator
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
