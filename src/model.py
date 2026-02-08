import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = self._block(2048, 1024)
        self.up2 = self._block(1024, 512)
        self.up3 = self._block(512, 256)
        self.up4 = self._block(256, 64)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        x = feats[0]
        x = self.up1(x) + feats[1]
        x = self.up2(x) + feats[2]
        x = self.up3(x) + feats[3]
        x = self.up4(x)
        x = self.up5(x)
        return torch.sigmoid(self.final(x))


class BuildingSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
