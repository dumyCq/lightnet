#
#   Darknet Tiny YOLOv3 model
#

from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn
import torch.nn.functional as F

__all__ = ['TinyYoloV3']
class TinyYoloV3(lnn.module.Darknet):
    """ Tiny Yolo v3 implementation :cite:`yolo_v3`.

    Args:
        num_classes (Number, optional): Number of classes; Default **30**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Tiny yolo v3 anchors**

    Attributes:
        self.stride: Subsampling factor of the network (input dimensions should be a multiple of this number)
        self.remap_darknet: Remapping rules for weights from the `~lightnet.models.Darknet` model.
	
    Note:
        Unlike YoloV2, the anchors here are defined as multiples of the input dimensions and not as a multiple of the output dimensions!
        The anchor list also has one more dimension than the one from YoloV2, in order to differentiate which anchors belong to which stride.

    """
    stride = (16,8)

    """
    remap_darknet53 = [
        (r'^layers.([a-w]_)',   r'extractor.\1'),   # Residual layers
        (r'^layers.(\d_)',      r'extractor.\1'),   # layers 1, 2, 5
        (r'^layers.([124]\d_)', r'extractor.\1'),   # layers 10, 27, 44
    ]
	
    """
    
    remap_darknet19 = [
        (r'^layers.0.([1-9]_)', r'layers.0.\1'),    #layers 1-9
    ]

    def __init__(self, num_classes=20, input_channels=3, anchors=[[(81, 82), (135, 169), (344, 319)], [(10, 14), (23,27), (37, 58)]]):

        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable) and not isinstance(anchors[0][0], Iterable):
            raise TypeError('Anchors need to be a 3D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.anchors = []   # YoloV3 defines anchors as a multiple of the input dimensions of the network as opposed to the output dimensions
        for i, s in enumerate(self.stride):
            self.anchors.append([(a[0] / s, a[1] / s) for a in anchors[i]])

        # Network
        self.extractor = nn.ModuleList([
            # Sequence 0 : input = input_channels
            nn.Sequential(
                OrderedDict([
                    ('1_convbatch',lnn.layer.Conv2dBatchReLU(input_channels,16,3,1,1)),
                    ('2_maxpool',nn.MaxPool2d(2,2)),
                    ('3_convbatch', lnn.layer.Conv2dBatchReLU(16,32, 3, 1, 1)),
                    ('4_maxpool', nn.MaxPool2d(2, 2)),
                    ('5_convbatch', lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                    ('6_maxpool', nn.MaxPool2d(2, 2)),
                    ('7_convbatch', lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                    ('8_maxpool', nn.MaxPool2d(2, 2)),
                    ('9_convbatch', lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ])
            ),

            # Sequence 1 : input = 9_convbatch
            nn.Sequential(
                OrderedDict([
                    ('10_maxpool', nn.MaxPool2d(2, 2)),
                    ('11_convbatch', lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('12_maxpool', lnn.layer.PaddedMaxPool2d(2, 1, (0 , 1, 0, 1))),
                    ('13_convbatch', lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('14_convbatch', lnn.layer.Conv2dBatchReLU(1024, 256, 1, 1, 0)),
                ])
            ),
        ])

        self.detector = nn.ModuleList([
            # Sequence 0 : input = 14_convbatch
            nn.Sequential(
                OrderedDict([
                    ('15_convbatch', lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('16_convbatch', lnn.layer.Conv2dBatchReLU(512, 255, 1, 1, 0, relu=lambda: nn.Identity())),
                    ('17_yolo', nn.Conv2d(255,len(self.anchors[0])*(5+self.num_classes),1,1,0)),
                ])
            ),

            # Sequence 1 : input = 14_convbatch
            nn.Sequential(
                OrderedDict([
                    ('18_convbatch', lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('19_upsample', nn.Upsample(scale_factor=2,mode='nearest')),
                ])
            ),

            #sequence 2 : input = 19_upsample + 8_maxpool
            nn.Sequential(
                OrderedDict([
                    ('20_convbatch', lnn.layer.Conv2dBatchReLU(128+256, 256, 3, 1, 1)),
                    ('21_convbatch', lnn.layer.Conv2dBatchReLU(256, 255, 1, 1, 0, relu=lambda: nn.Identity())),
                    ('22_yolo', nn.Conv2d(255,len(self.anchors[1])*(5+self.num_classes),1,1,0)),
                ])
            ),
        ])

    def forward(self,x):
        out = [None,None]
        # Feature extractor
        inter_features_0 = self.extractor[0](x) # 8_maxpool
        inter_features_1 = self.extractor[1](inter_features_0) # 13_convbatch

        # detector 0
        out[0] = self.detector[0](inter_features_1) # 17_yolo

        #detector 1
        extra_features = self.detector[1](inter_features_1) # 19_upsample
        out[1] = self.detector[2](torch.cat((extra_features,inter_features_0),1)) # 22_yolo

        return out
