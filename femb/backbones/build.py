import torchvision
import torch

from .networks import *

def build_backbone(backbone='iresnet50', embed_dim=512, pretrained=False):

    if backbone == 'iresnet18':
        assert not pretrained
        return iresnet.iresnet18(num_classes=embed_dim)

    if backbone == 'iresnet34':
        assert not pretrained
        return iresnet.iresnet34(num_classes=embed_dim)

    if backbone == 'iresnet50':
        assert not pretrained
        return iresnet.iresnet50(num_classes=embed_dim)

    if backbone == 'alexnet':
        backbone = torchvision.models.alexnet(pretrained=pretrained)

        classifier = [
         torch.nn.Dropout(p=0.5, inplace=False),
         torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
         torch.nn.ReLU(inplace=True),
         torch.nn.Dropout(p=0.5, inplace=False),
         torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
         torch.nn.ReLU(inplace=True),
         torch.nn.Linear(in_features=4096, out_features=embed_dim, bias=True),
        ]

        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]) + classifier)
        return backbone

    if backbone == 'vgg16':
        backbone = torchvision.models.vgg16(pretrained=pretrained)

        classifier = [
         torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
         torch.nn.ReLU(inplace=True),
         torch.nn.Dropout(p=0.5, inplace=False),
         torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
         torch.nn.ReLU(inplace=True),
         torch.nn.Dropout(p=0.5, inplace=False),
         torch.nn.Linear(in_features=4096, out_features=embed_dim, bias=True)
        ]

        # remove everything after last conv layer
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]) + classifier)
        return backbone

    if backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)

        classifier = [
         torch.nn.Dropout(p=0.2, inplace=False),
         torch.nn.Linear(in_features=1280, out_features=embed_dim, bias=True)
        ]

        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]) + classifier)
        return backbone


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
