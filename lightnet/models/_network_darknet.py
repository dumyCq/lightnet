#
#   Darknet Darknet19 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['Darknet']


class Darknet(lnn.module.Darknet):
    """ `Darknet`_ implementation with pytorch.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)

    .. _Darknet: https://github.com/pjreddie/darknet/blob/master/cfg/darknet.cfg
    """
    stride = 32

    def __init__(self, num_classes=1000, input_channels=3):
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Network
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 16, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 32, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('6_max',           nn.MaxPool2d(2, 2)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('10_max',          nn.MaxPool2d(2, 2)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('14_avgpool',      lnn.layer.GlobalAvgPool2d())
                ('15_conv',         nn.Conv2d(1024, num_classes, 1, 1, 0)),
            ])
        )
