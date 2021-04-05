import torch


class LinearHeader(torch.nn.Module):
    """ Linear Header class"""

    def __init__(self, in_features, out_features):
        super(LinearHeader, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, input, label):
        return self.linear(input)
