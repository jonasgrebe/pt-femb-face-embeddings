import torchvision

from modules.networks import iresnet

def build_backbone(backbone, embed_dim):

    if backbone == 'iresnet18':
        return iresnet.iresnet18(pretrained=True, num_classes=embed_dim)

    return backbone
