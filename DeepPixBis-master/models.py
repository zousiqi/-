r'''
DeepPix Implementation based on
    https://publications.idiap.ch/downloads/papers/2019/George_ICB2019.pdf
'''
import torch
from torch.nn import Module, Linear, Conv2d, Sigmoid

# Load Pretrained DenseNet121 Network
DenseNet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)


class DeepPix(Module):

    def __init__(self):
        super(DeepPix, self).__init__()

        self.conv0 = DenseNet.features.conv0
        self.norm0 = DenseNet.features.norm0
        self.relu0 = DenseNet.features.relu0
        self.pool0 = DenseNet.features.pool0
        self.denseblock1 = DenseNet.features.denseblock1
        self.transition1 = DenseNet.features.transition1
        self.denseblock2 = DenseNet.features.denseblock2
        self.transition2 = DenseNet.features.transition2

        for param in self.conv0.parameters():
            param.requires_grad = True

        for param in self.norm0.parameters():
            param.requires_grad = True

        for param in self.relu0.parameters():
            param.requires_grad = True

        for param in self.pool0.parameters():
            param.requires_grad = True

        for param in self.denseblock1.parameters():
            param.requires_grad = True

        for param in self.transition1.parameters():
            param.requires_grad = True

        for param in self.denseblock2.parameters():
            param.requires_grad = True

        for param in self.transition2.parameters():
            param.requires_grad = True

        self.conv1x1 = Conv2d(256, 1, kernel_size=1, stride=1)

        self.sigmoid1 = Sigmoid()

        self.linear1 = Linear(196, 1)

        self.sigmoid2 = Sigmoid()

    def forward(self, x):

        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.conv1x1(x)
        x = self.sigmoid1(x)
        y = x.view(x.shape[0], -1)
        y = self.linear1(y)
        y = self.sigmoid2(y)

        return y, x

if __name__ == "__main__":
    
    model = DeepPix()

    x = torch.rand(14, 3, 224, 224)

    out = model(x)

    print(out[1].shape)


        