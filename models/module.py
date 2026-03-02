import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden_channel,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(
            hidden_channel, out_channel, kernel_size, stride, padding
        )
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Encoder, self).__init__()
        self.dc1 = DoubleConv(in_channel, 64, 64, kernel_size, stride, padding)
        self.dc2 = DoubleConv(64, 128, 128, kernel_size, stride, padding)
        self.dc3 = DoubleConv(128, 256, 256, kernel_size, stride, padding)
        self.dc4 = DoubleConv(
            256, out_channel // 2, out_channel, kernel_size, stride, padding
        )
        # self.dc5 = DoubleConv(
        #     512, out_channel, out_channel, kernel_size, stride, padding
        # )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        # self.pool4 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(self.pool1(x1))
        x3 = self.dc3(self.pool2(x2))
        x4 = self.dc4(self.pool3(x3))
        # x5 = self.dc5(self.pool4(x))
        return x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Decoder, self).__init__()
        self.dc1 = DoubleConv(in_channel, 128, 256, kernel_size, stride, padding)
        self.dc2 = DoubleConv(256, 64, 128, kernel_size, stride, padding)
        self.dc3 = DoubleConv(128, 32, 64, kernel_size, stride, padding)
        self.conv = nn.Conv2d(32, out_channel, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, x1, x2, x3):
        x = torch.cat([self.up1(x), x1], dim=1)
        x = self.dc1(x)
        x = torch.cat([self.up2(x), x2], dim=1)
        x = self.dc2(x)
        x = torch.cat([self.up3(x), x3], dim=1)
        x = self.dc3(x)
        x = self.conv(x)
        return x


class ResolutionUNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ResolutionUNet, self).__init__()
        self.encoder = Encoder(in_channel, 512, kernel_size, stride, padding)
        self.decoder = Decoder(512, out_channel, kernel_size, stride, padding)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        x = F.tanh(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = ResolutionUNet(3, 3, 3, 1, 1)
    summary(model, (5, 3, 32, 64))
