# Function：model.py
# Author：MIVRC
# Time：2018.2.1

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
from collections import OrderedDict


# --------------------------MSRB------------------------------- #

class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)

        output = self.confusion(output)
        output = torch.add(output, identity_data)
        return output


# --------------------------Model------------------------------- #

class MSRN(nn.Module):
    def __init__(self, scale, num_of_block=18, features=64):
        super(MSRN, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        out_features = (num_of_block + 1) * features
        self.features = nn.Sequential(OrderedDict([
            ('msrb0', MSRB_Block())
        ]))
        for i in range(num_of_block - 1):
            block = MSRB_Block()
            self.features.add_module('msrb%d' % (i + 1), block)

        self.bottle = nn.Conv2d(in_channels=out_features, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Conv2d(in_channels=64, out_channels=64 * scale * scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.subpixle = nn.PixelShuffle(scale)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv_input(x)

        out_seq = [out]
        for m in self.features:
            out = m(out)
            out_seq.append(out)

        out = torch.cat(out_seq, 1)
        out = self.bottle(out)
        out = self.subpixle(self.conv_up(out))
        out = self.conv_output(out)
        return out
