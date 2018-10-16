from torch.nn import Linear, Sequential, ReLU, LeakyReLU, BatchNorm2d, Conv2d, AvgPool2d, Softmax, Sigmoid, Tanh, \
    ConvTranspose2d, Module, BatchNorm1d


# Residual block
# Input feature map
# 3 x 3 conv. out channels RELU stride 2 pad 1
# 3 x 3 conv. out channels RELU stride 1 pad 1
# skip connection output = input + residual
# RELU
class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU()
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU()
        )
        self.relu = ReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        output = x + res
        output = self.relu(output)
        return output


class Down(Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding):
        super(Down, self).__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU()
        )
        self.relu = ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        return output


class Up(Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding):
        super(Up, self).__init__()
        self.upconv = Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU()
        )
        self.relu = ReLU()

    def forward(self, x):
        output = self.upconv(x)
        output = self.relu(output)
        return output


# Encoder
# Input 32 x 32 images
# 3 x 3 conv. 64 RELU stride 2 pad 1
# 3 x 3 residual block 64
# 3 x 3 down sampling residual block 128 stride 2
# 3 x 3 down sampling residual block 256 stride 2
# 3 x 3 down sampling residual block 512 stride 2 -- I've removed third downsampling, because MNIST 28 x 28 is too small
# 4 x 4 avg. pooling stride 1
# FC. 2 x code size BN. RELU
# FC. code size Linear

class Encoder(Module):
    def __init__(self, code_size=8, input_image_channels=1, input_image_size=28):
        super(Encoder, self).__init__()

        self.input_image_size = input_image_size

        self.conv1 = Conv2d(in_channels=input_image_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(in_channels=64, out_channels=64)

        self.down_sampling1 = Sequential(
            Down(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(in_channels=128, out_channels=128)
        )
        self.down_sampling2 = Sequential(
            Down(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(in_channels=256, out_channels=256)
        )

        self.pooling = AvgPool2d(kernel_size=2, stride=1)  # kernel size 2 for MNIST
        self.fc1 = Sequential(
            Linear(256, 2 * code_size),
            BatchNorm1d(2 * code_size),
            ReLU()
        )
        self.fc2 = Linear(2 * code_size, code_size)

    def forward(self, x):
        # we should have input image input_image_size x input_image_size
        assert x.shape[-1] == self.input_image_size
        assert x.shape[-2] == self.input_image_size

        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.down_sampling1(x)
        x = self.down_sampling2(x)

        x = self.pooling(x).squeeze(dim=-1).squeeze(dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)

        return x


# Decoder
# Input latent code ∈ Rcode size
# 4 x 4 upconv. 512 BN. RELU stride 1
# 4 x 4 up sampling residual block 256 stride 2
# 4 x 4 up sampling residual block 128 stride 2
# 4 x 4 up sampling residual block 64 stride 2 -- I've removed third upsampling for MNIST
# 3 x 3 conv. image channels Tanh
class Decoder(Module):
    def __init__(self, code_size=8, out_image_channels=1):
        super(Decoder, self).__init__()
        self.code_size = code_size

        # kernel size 7 for MNIST
        self.upconv = ConvTranspose2d(in_channels=code_size, out_channels=512, kernel_size=7, stride=2)

        self.up_sampling1 = Sequential(
            Up(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=0),
            ResidualBlock(in_channels=256, out_channels=256)
        )
        self.up_sampling2 = Sequential(
            Up(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0),
            ResidualBlock(in_channels=128, out_channels=128)
        )

        self.conv = Sequential(
            # kernel size 7 for MNIST
            Conv2d(in_channels=128, out_channels=out_image_channels, kernel_size=7),
            Tanh()
        )

    def forward(self, x):
        assert x.shape[1] == self.code_size

        x = self.upconv(x)

        x = self.up_sampling1(x)
        x = self.up_sampling2(x)
        x = self.conv(x)

        return x


# Code Generator
# Input noise ∈ Rnoise size
# FC. 2 x noise size BN. RELU
# FC. latent code size BN. Linear
class CodeGenerator(Module):
    def __init__(self, code_size=8, noise_size=8):
        super(CodeGenerator, self).__init__()
        self.noise_size = noise_size
        self.fc1 = Sequential(
            Linear(noise_size, 2 * noise_size),
            BatchNorm1d(2 * noise_size),
            ReLU()
        )
        self.fc2 = Sequential(
            Linear(2 * noise_size, code_size),
            BatchNorm1d(code_size)
        )

    def forward(self, x):
        # we should have input of shape noise_size
        assert x.shape[1] == self.noise_size
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return x


# Image Discriminator D/Q
# Input 32 x 32 images
# 4 x 4 conv. 64 LRELU stride 2 pad 1
# 4 x 4 conv. 128 BN LRELU stride 2 pad 1
# 4 x 4 conv. 256 BN LRELU stride 2 pad 1
# FC. 1000 LRELU
# FC 1 Sigmoid for D
# FC 10 Softmax for Q
class ImageDiscriminator(Module):
    def __init__(self, input_image_channels=1, input_image_size=28):
        super(ImageDiscriminator, self).__init__()

        self.input_image_size = input_image_size

        self.conv1 = Sequential(
            Conv2d(in_channels=input_image_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            LeakyReLU()
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(num_features=128),
            LeakyReLU()
        )
        self.conv3 = Sequential(  # kernel_size=7, stride=3 for MNIST
            Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=3, padding=1),
            BatchNorm2d(num_features=256),
            LeakyReLU()
        )
        self.fc = Sequential(
            Linear(256, 1000),
            LeakyReLU()
        )
        self.fc1 = Sequential(
            Linear(1000, 1),
            Sigmoid()
        )
        self.fc10 = Sequential(
            Linear(1000, 10),
            Softmax()
        )

    def forward(self, x):
        # we should have input image input_image_size x input_image_size
        assert x.shape[-1] == self.input_image_size
        assert x.shape[-2] == self.input_image_size
        x = self.conv1(x)
        x = self.conv2(x)
        map_F = self.conv3(x)
        x = map_F.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        x1 = self.fc1(x)
        x1 = x1.view(x1.shape[0], x1.shape[1], 1, 1)
        x10 = self.fc10(x)

        return x1, x10, map_F


# Code Discriminator
# Input latent code
# FC 1000 LRELU
# FC 500 LRELU
# FC 200 LRELU
# FC 1 Sigmoid
class CodeDiscriminator(Module):
    def __init__(self, code_size=8):
        super(CodeDiscriminator, self).__init__()
        self.code_size = code_size
        self.fc1 = Sequential(
            Linear(code_size, 1000),
            LeakyReLU()
        )
        self.fc2 = Sequential(
            Linear(1000, 500),
            LeakyReLU()
        )
        self.fc3 = Sequential(
            Linear(500, 200),
            LeakyReLU()
        )
        self.fc = Sequential(
            Linear(200, 1),
            Sigmoid()
        )

    def forward(self, x):
        # we should have input of shape code_size
        assert x.shape[1] == self.code_size

        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.fc(x)

        return x
