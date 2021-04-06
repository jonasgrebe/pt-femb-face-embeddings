from .arcmargin import ArcMarginHeader


class SphereFaceHeader(ArcMarginHeader):
    """ SphereFaceHeader class"""

    def __init__(self, in_features, out_features, m=4):
        super(SphereFaceHeader, self).__init__(in_features=in_features, out_features=out_features, s=1, m1=m, m2=0, m3=0)
