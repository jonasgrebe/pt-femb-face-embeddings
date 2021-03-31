import torch

class FaceLossHeader(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(FaceLossHeader, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input, label):
        return self.linear(input)


class LinearHeader(FaceLossHeader):
    """ Linear Header class"""

    def __init__(self, in_features, out_features):
        super(LinearHeader, self).__init__(in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, input, label):
        return self.linear(input)


class ArcFaceHeader(FaceLossHeader):
    """ ArcFaceHeader class"""

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceHeader, self).__init__(in_features, out_features)

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


class MagFaceHeader(FaceLossHeader):
    """ MagFaceHeader class"""

    def __init__(self, in_features, out_features, s=64.0, m=0.5, l_a, u_m, l_a, u_m, lambda_g):
        super(MagFaceHeader, self).__init__(in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m

        self.lambda_g = lambda_g

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.normalize = torch.nn.functional.normalize


    def compute_m(self, a):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_m


    def compute_g(self, a):
        return torch.mean(1 / self.u_a**2 * a + 1 / a)


    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
        self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        logits = self.linear(self.normalize(input))

        # difference compared to arcface
        a = torch.norm(x, dim=1, keepdims=True).clamp(l_a, u_a)
        m = self.compute_m(a)
        g = self.compute_g(a)

        # apply arccos to get theta
        theta = torch.acos(logits)

        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(theta + m)

        # derive one-hot encoding for label
        one_hot = torch.zeros(logits.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # build the output logits
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        # feature re-scaling
        output *= self.s

        return output + lambda_g * g
