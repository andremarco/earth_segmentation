import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel


class ICNet(SegBaseModel):
    """Image Cascade Network"""
    
    def __init__(self, n_channels=3, n_classes=19, backbone='resnet50', pretrained_base=True):
        super(ICNet, self).__init__(n_classes,backbone, input_size=n_channels, pretrained_base=pretrained_base)
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(in_channels=n_channels, out_channels=32, kernel_size=3, stride=2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        
        self.ppm = PyramidPoolingModule()

        self.head = _ICHead(n_classes)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])
        
    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.base_forward(x_sub2)
        
        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.base_forward(x_sub4)
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)
        
        outputs = self.head(x_sub1, x_sub2, x_sub4)
        
        return tuple(outputs)

class PyramidPoolingModule(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat
    
class _ICHead(nn.Module):
    def __init__(self, n_classes, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(low_channels=128, high_channels=64, out_channels=128, n_classes=n_classes, norm_layer=norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, n_classes, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2d(128, n_classes, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        # cff sub 4 and sub 2
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        # cff cff 24 and sub 1
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x4 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x4)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, n_classes, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, n_classes, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


if __name__ == '__main__':
    pass