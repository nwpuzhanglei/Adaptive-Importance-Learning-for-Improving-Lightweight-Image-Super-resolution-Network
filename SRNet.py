import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self, Cn=64):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=Cn, out_channels=Cn, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class acSRNet(nn.Module):
    def __init__(self, Cn):
        super(acSRNet, self).__init__()

        self.Cn = Cn
        self.input = nn.Conv2d(in_channels=1, out_channels=self.Cn, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18, self.Cn)

        self.output = nn.Conv2d(in_channels=self.Cn, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                 m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer, cn):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(cn))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

