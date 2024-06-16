from torch import flatten, sigmoid
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool2d, Linear
from torchvision.models.resnet import BasicBlock


class ResNet32iCaRL(Module):
    """This is an implementation of ResNet32 adapted according to the original iCaRL paper. The building blocks used to
     construct the neural network are the original ones taken from the PyTorch module. The style of constructing the
     model is also borrowed from the original implementation of ResNets in the aforementioned module."""
    def __init__(self, num_classes=10):
        super(ResNet32iCaRL, self).__init__()

        self.in_channels = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.relu = ReLU(inplace=True)
        self.layers = Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 32, stride=2, downsample=Sequential(
                    Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                    BatchNorm2d(32)
                )
            ),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 64, stride=2, downsample=Sequential(
                    Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                    BatchNorm2d(64)
                )
            ),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)

        if return_features:
            x = x.mean(dim=[2, 3])

            return x

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return sigmoid(x)
