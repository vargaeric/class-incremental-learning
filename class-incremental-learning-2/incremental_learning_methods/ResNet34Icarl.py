from torchvision.models import resnet34
from torch import sigmoid
from torch.nn import Module, Sequential, AdaptiveAvgPool2d, Flatten, Linear

class ResNet34Icarl(Module):
    def __init__(self, classes_nr):
        super(ResNet34Icarl, self).__init__()

        base_model = resnet34(weights=None)

        self.feature_extractor = Sequential(*(list(base_model.children())[:-2]))
        self.pool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = Flatten()
        self.fc = Linear(512, classes_nr)

    def forward(self, x, return_features=False):
        x = self.feature_extractor(x)

        if return_features:
            return x.squeeze()

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return sigmoid(x)
