import torch


class ArcFaceHeader(torch.nn.Module):
    """ ArcFaceHeader class"""

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceHeader, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.normalize = torch.nn.functional.normalize


    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
        self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        logits = self.linear(self.normalize(input))

        # apply arccos to get theta
        theta = torch.acos(logits)
        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(theta + self.m)

        # derive one-hot encoding for label
        one_hot = torch.zeros(logits.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # build the output logits
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        # feature re-scaling
        output *= self.s

        return output
