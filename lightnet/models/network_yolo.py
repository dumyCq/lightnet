#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['Yolo']


class Yolo(lnn.Darknet):
    """ `Yolo v2`_ implementation with pytorch.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (dict, optional): Dictionary containing `num` and `values` properties with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Yolo v2: https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg
    """
    def __init__(self, num_classes=20, weights_file=None, conf_thresh=.25, nms_thresh=.4, input_channels=3, anchors=dict(num=5, values=[1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071])):
        """ Network initialisation """
        super(Yolo, self).__init__()

        # Parameters
        self.num_classes = num_classes
        self.num_anchors = anchors['num']
        self.anchors = anchors['values']
        self.reduction = 32             # input_dim/output_dim

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchLeaky(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 64, 1, 1, 0)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, self.num_anchors*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        self.load_weights(weights_file)
        self.loss = lnn.RegionLoss(self) 
        self.postprocess = lnd.GetBoundingBoxes(self, conf_thresh, nms_thresh)

    def _forward(self, x):
        outputs = []
    
        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        # Route : layers=-9
        outputs.append(self.layers[2](outputs[0]))
        # Route : layers=-1,-4
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        return out