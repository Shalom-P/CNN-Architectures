import torch
import torch.nn as nn
from torch import nn, Tensor

###stochastic depth layer
##################################################
#only the stochastic depth layer has been copied from the orirginal pytorch implementation
######################################################
def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s
#####################################################################################################################
##################################################################################################################
"""
Implements the efficientnet b0 architecture from 
'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
https://arxiv.org/abs/1905.11946
"""

######################################################################################################################
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,activation=True,groups=1):
        super(Conv, self).__init__()
        # stride = kernel_size
        if activation:
            self.conv = torch.nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=False,groups=groups,stride=stride,padding=padding),
                nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True)
            )
        else:
            self.conv = torch.nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=False,groups=groups,stride=stride),
                nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    def forward(self,x):
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        # Excitation operation
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.excitation(x)
        return x.mul_(y)



class MbConv1(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1):
        super(MbConv1,self).__init__()   
        self.mbconv1 = nn.Sequential(
            Conv(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,groups=in_channels,stride=stride,padding=1),
            SEBlock(in_channels=in_channels,reduction_ratio=4),
            Conv(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=1,activation=False)
        ) 

    def forward(self,x_in):
        x = self.mbconv1(x_in)
        return x


class MbConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super(MbConv,self).__init__()   
        self.mbconv = nn.Sequential(
            Conv(in_channels=in_channels,out_channels=in_channels*6,kernel_size=1,stride=1),
            Conv(in_channels=in_channels*6,out_channels=in_channels*6,kernel_size=kernel_size,groups=in_channels*6,stride=stride,padding=padding),
            SEBlock(in_channels=in_channels*6,reduction_ratio=24),
            Conv(in_channels=in_channels*6,out_channels=out_channels,stride=1,kernel_size=1,activation=False)
        ) 

    def forward(self,x_in):
        x = self.mbconv(x_in)
        return x
#############################################################################################
# efficientnetb0 without stochasticdepth
#############################################################################################
class efficientnet_b0_without_stochastic(nn.Module):
    def __init__(self,num_classes=2):
        super(efficientnet_b0,self).__init__()
        self.conv1 = Conv(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.part1 = MbConv1(32,16,kernel_size=3)
        # self.stoc1 = StochasticDepth(0.0,"row")
        self.mbconv_1_0 = MbConv(16,24,kernel_size=3,stride=2,padding=1)
        # self.stoc2 = StochasticDepth(0.0125,"row")
        self.mbconv_1_1 = MbConv(24,24,kernel_size=3,padding=1)
        # self.stoc3 = StochasticDepth(0.025,"row")
        self.mbconv_2_0 = MbConv(24,40,kernel_size=5,stride=2,padding=2)
        # self.stoc4 = StochasticDepth(0.037500000000000006,"row")
        self.mbconv_2_1 = MbConv(40,40,kernel_size=5,padding=2)
        # self.stoc5 = StochasticDepth(0.05,"row")
        self.mbconv_3_0 = MbConv(40,80,kernel_size=3,stride=2,padding=1)
        # self.stoc6 = StochasticDepth(0.0625,"row")
        self.mbconv_3_1 = MbConv(80,80,kernel_size=3,padding=1)
        # self.stoc7 = StochasticDepth(0.07500000000000001,"row")
        self.mbconv_3_2 = MbConv(80,80,kernel_size=3,padding=1)
        # self.stoc8 = StochasticDepth(0.08750000000000001,"row")
        self.mbconv_4_0 = MbConv(80,112,kernel_size=5,padding=2)
        # self.stoc9 = StochasticDepth(0.1,"row")
        self.mbconv_4_1 = MbConv(112,112,kernel_size=5,padding=2)
        # self.stoc10 = StochasticDepth(0.1125,"row")
        self.mbconv_4_2 = MbConv(112,112,kernel_size=5,padding=2)
        # self.stoc11 = StochasticDepth(0.125,"row")
        self.mbconv_5_0 = MbConv(112,192,kernel_size=5,padding=2,stride=2)
        # self.stoc12 = StochasticDepth(0.1375,"row")
        self.mbconv_5_1 = MbConv(192,192,kernel_size=5,padding=2)
        # self.stoc13 = StochasticDepth(0.15000000000000002,"row")
        self.mbconv_5_2 = MbConv(192,192,kernel_size=5,padding=2)
        # self.stoc14 = StochasticDepth(0.1625,"row")
        self.mbconv_5_3 = MbConv(192,192,kernel_size=5,padding=2)
        # self.stoc15 = StochasticDepth(0.17500000000000002,"row")
        self.mbconv_6_0 = MbConv(192,320,kernel_size=3,padding=1)
        # self.stoc16 = StochasticDepth(0.1875,"row")
        self.conv2 = Conv(320,1280,kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(1280,out_features=num_classes)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.part1(x)
        x = self.mbconv_1_0(x)
        x = self.mbconv_1_1(x)+x
        x = self.mbconv_2_0(x)
        x = self.mbconv_2_1(x)+x
        x = self.mbconv_3_0(x)
        x = self.mbconv_3_1(x)+x
        x = self.mbconv_3_2(x)+x
        x = self.mbconv_4_0(x)
        x = self.mbconv_4_1(x)+x
        x = self.mbconv_4_2(x)+x
        x = self.mbconv_5_0(x)
        x = self.mbconv_5_1(x)+x
        x = self.mbconv_5_2(x)+x
        x = self.mbconv_5_3(x)+x
        x = self.mbconv_6_0(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

#############################################################################################
# efficientnetb0 with stochasticdepth
#############################################################################################

class efficientnet_b0(nn.Module):
    def __init__(self,num_classes=2):
        super(efficientnet_b0,self).__init__()
        self.conv1 = Conv(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.part1 = MbConv1(32,16,kernel_size=3)
        self.stoc1 = StochasticDepth(0.0,"row")
        self.mbconv_1_0 = MbConv(16,24,kernel_size=3,stride=2,padding=1)
        self.stoc2 = StochasticDepth(0.0125,"row")
        self.mbconv_1_1 = MbConv(24,24,kernel_size=3,padding=1)
        self.stoc3 = StochasticDepth(0.025,"row")
        self.mbconv_2_0 = MbConv(24,40,kernel_size=5,stride=2,padding=2)
        self.stoc4 = StochasticDepth(0.037500000000000006,"row")
        self.mbconv_2_1 = MbConv(40,40,kernel_size=5,padding=2)
        self.stoc5 = StochasticDepth(0.05,"row")
        self.mbconv_3_0 = MbConv(40,80,kernel_size=3,stride=2,padding=1)
        self.stoc6 = StochasticDepth(0.0625,"row")
        self.mbconv_3_1 = MbConv(80,80,kernel_size=3,padding=1)
        self.stoc7 = StochasticDepth(0.07500000000000001,"row")
        self.mbconv_3_2 = MbConv(80,80,kernel_size=3,padding=1)
        self.stoc8 = StochasticDepth(0.08750000000000001,"row")
        self.mbconv_4_0 = MbConv(80,112,kernel_size=5,padding=2)
        self.stoc9 = StochasticDepth(0.1,"row")
        self.mbconv_4_1 = MbConv(112,112,kernel_size=5,padding=2)
        self.stoc10 = StochasticDepth(0.1125,"row")
        self.mbconv_4_2 = MbConv(112,112,kernel_size=5,padding=2)
        self.stoc11 = StochasticDepth(0.125,"row")
        self.mbconv_5_0 = MbConv(112,192,kernel_size=5,padding=2,stride=2)
        self.stoc12 = StochasticDepth(0.1375,"row")
        self.mbconv_5_1 = MbConv(192,192,kernel_size=5,padding=2)
        self.stoc13 = StochasticDepth(0.15000000000000002,"row")
        self.mbconv_5_2 = MbConv(192,192,kernel_size=5,padding=2)
        self.stoc14 = StochasticDepth(0.1625,"row")
        self.mbconv_5_3 = MbConv(192,192,kernel_size=5,padding=2)
        self.stoc15 = StochasticDepth(0.17500000000000002,"row")
        self.mbconv_6_0 = MbConv(192,320,kernel_size=3,padding=1)
        self.stoc16 = StochasticDepth(0.1875,"row")
        self.conv2 = Conv(320,1280,kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(1280,out_features=num_classes)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.stoc1(self.part1(x))
        x = self.stoc2(self.mbconv_1_0(x))
        x = self.stoc3(self.mbconv_1_1(x))+x
        x = self.stoc4(self.mbconv_2_0(x))
        x = self.stoc5(self.mbconv_2_1(x))+x
        x = self.stoc6(self.mbconv_3_0(x))
        x = self.stoc7(self.mbconv_3_1(x))+x
        x = self.stoc8(self.mbconv_3_2(x))+x
        x = self.stoc9(self.mbconv_4_0(x))
        x = self.stoc10(self.mbconv_4_1(x))+x
        x = self.stoc11(self.mbconv_4_2(x))+x
        x = self.stoc12(self.mbconv_5_0(x))
        x = self.stoc13(self.mbconv_5_1(x))+x
        x = self.stoc14(self.mbconv_5_2(x))+x
        x = self.stoc15(self.mbconv_5_3(x))+x
        x = self.stoc16(self.mbconv_6_0(x))
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


######################################################################################################################
"""
#Implements the growing cosine unit from 
'Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks'
#https://arxiv.org/abs/2108.12943
"""
######################################################################################################################
###growing cosine activation function
class GCU(nn.Module):
    def __init__(self,inplace=False):
        super(GCU,self).__init__()  
        self.inplace = inplace
    def forward(self,x):
        if self.inplace:
            return x*torch.cos_(x)
        else:
            return x*torch.cos(x)


if __name__=='__main__':
    
    mod = efficientnet_b0(num_classes=8)
    inn = torch.randn((7,3,224,224))
    out = mod(inn)
    print(out.size())