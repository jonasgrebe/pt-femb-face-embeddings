import torch
from backbones import build_backbone
from headers import LinearHeader, ArcFaceHeader
from face import FaceRecognitionModel
from data import FaceDataset

def main():
    embed_dim = 512

    train_dataset = FaceDataset("lfw")
    n_classes = train_dataset.get_n_identities()

    # build backbone, header and ce loss
    backbone = build_backbone(backbone="resnet-50", embed_dim=embed_dim)
    header = ArcFaceHeader(in_features=embed_dim, out_features=n_classes)
    loss = torch.nn.CrossEntropyLoss()

    # join parameter sets of backbone and header
    # params = params = list(backbone.parameters())
    # params.extend(list(header.parameters()))

    # create the face recognition model wrapper
    face_model = FaceRecognitionModel(backbone=backbone, header=header, loss=loss)

    # fit the face recognition model to the dataset
    face_model.fit(
        dataset=train_dataset,
        epochs=10,
        batch_size=train_batch_size,
        optimizer='SGD',
        preprocessing='center_crop:128x128',
        evaluator=None
    )

    embedding1, embedding2 = face_model.encode(imgs=[img1, img2])


if __name__ == '__main__':

    main()
