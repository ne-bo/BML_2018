import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# Residual block
# Input feature map
# 3 x 3 conv. out channels RELU stride 2 pad 1
# 3 x 3 conv. out channels RELU stride 1 pad 1
# skip connection output = input + residual
# RELU
class ResidualBlock(BaseModel):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,
                                             out_channels=out_channels,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        output = x + res
        output = self.relu(output)
        return output


# Encoder
# Input 32 x 32 images
# 3 x 3 conv. 64 RELU stride 2 pad 1
# 3 x 3 residual block 64
# 3 x 3 down sampling residual block 128 stride 2
# 3 x 3 down sampling residual block 256 stride 2
# 3 x 3 down sampling residual block 512 stride 2
# 4 x 4 avg. pooling stride 1
# FC. 2 x code size BN. RELU
# FC. code size Linear

class Encoder(BaseModel):
    def __init__(self, code_size=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(in_channels=64, out_channels=64)

        self.down_sampling1 = nn.Sequential(DownSampling(), ResidualBlock(in_channels=128, out_channels=128))
        self.down_sampling2 = nn.Sequential(DownSampling(), ResidualBlock(in_channels=256, out_channels=256))
        self.down_sampling3 = nn.Sequential(DownSampling(), ResidualBlock(in_channels=512, out_channels=512))
        self.pooling = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc1 = nn.Sequential(nn.Linear(???, 2 * code_size), nn.BatchNorm2d(2 * code_size), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(2 * code_size, code_size)

    def forward(self, x):
        # we should have input image 32 x 32
        assert x.shape[-1] == 32
        assert x.shape[-2] == 32

        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.down_sampling1(x)
        x = self.down_sampling2(x)
        x = self.down_sampling3(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# Decoder
# Input latent code ∈ Rcode size
# 4 x 4 upconv. 512 BN. RELU stride 1
# 4 x 4 up sampling residual block 256 stride 2
# 4 x 4 up sampling residual block 128 stride 2
# 4 x 4 up sampling residual block 64 stride 2
# 3 x 3 conv. image channels Tanh


# Code Generator
# Input noise ∈ Rnoise size
# FC. 2 x noise size BN. RELU
# FC. latent code size BN. Linear
