import torch
from torch.nn import Module, Sequential, BatchNorm2d, Conv2d, ReLU, AdaptiveAvgPool2d, Linear


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def batch_norm(num_channels):
    """Batch normalization"""
    return BatchNorm2d(num_channels)


class ResidualBlock(Module):
    """Residual Block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = batch_norm(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = batch_norm(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet32(Module):
    """ResNet32 Model"""

    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
        self.in_channels = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = batch_norm(16)
        self.relu = ReLU(inplace=True)

        # Layer 1 - 16 channels, no stride change
        self.layer1 = Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16)
        )

        # Layer 2 - Increase to 32 channels, stride 2 for downsampling
        downsample1 = Sequential(
            Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            batch_norm(32)
        )
        self.layer2 = Sequential(
            ResidualBlock(16, 32, stride=2, downsample=downsample1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )

        # Layer 3 - Increase to 64 channels, stride 2 for downsampling
        downsample2 = Sequential(
            Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            batch_norm(64)
        )
        self.layer3 = Sequential(
            ResidualBlock(32, 64, stride=2, downsample=downsample2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if return_features:
            # Global Average Pooling to get shape (batch_size, 64)
            x = x.mean(dim=[2, 3])
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)  # Using sigmoid for final activation as per your model specifics
