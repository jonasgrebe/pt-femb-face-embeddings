import torchvision

from modules.networks import iresnet

def build_backbone(backbone, embed_dim):

    if backbone == 'iresnet18':
        return iresnet.iresnet18(num_classes=embed_dim)
    if backbone == 'iresnet34':
        return iresnet.iresnet34(num_classes=embed_dim)
    if backbone == 'iresnet50':
        return iresnet.iresnet50(num_classes=embed_dim)

    return backbone
