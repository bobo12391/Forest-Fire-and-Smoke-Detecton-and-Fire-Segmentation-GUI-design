import torch
import torch.nn as nn
from torch.nn import functional as F


# 注意力模块
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# 编码连续卷积层
def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return block


# 上采样过程中也反复使用了两层卷积的结构
double_conv = contracting_block

# 上采样反卷积模块
class expansive_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.block(x)
        return out


def final_block(in_channels, out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0)


class Unet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        # Encode
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = contracting_block(in_channels=64, out_channels=128)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = contracting_block(in_channels=128, out_channels=256)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = contracting_block(in_channels=256, out_channels=512)
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode5 = contracting_block(in_channels=512, out_channels=1024)

        # Decode
        self.conv_decode4 = expansive_block(1024, 512)
        self.att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.double_conv4 = double_conv(1024, 512)

        self.conv_decode3 = expansive_block(512, 256)
        self.att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.double_conv3 = double_conv(512, 256)

        self.conv_decode2 = expansive_block(256, 128)
        self.att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.double_conv2 = double_conv(256, 128)

        self.conv_decode1 = expansive_block(128, 64)
        self.att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.double_conv1 = double_conv(128, 64)

        self.final_layer = final_block(64, out_channel)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)
        encode_block5 = self.conv_encode5(encode_pool4)

        # Decode
        decode_block4 = self.conv_decode4(encode_block5)
        encode_block4 = self.att4(g=decode_block4, x=encode_block4)
        decode_block4 = torch.cat((encode_block4, decode_block4), dim=1)
        decode_block4 = self.double_conv4(decode_block4)

        decode_block3 = self.conv_decode3(decode_block4)
        encode_block3 = self.att3(g=decode_block3, x=encode_block3)
        decode_block3 = torch.cat((encode_block3, decode_block3), dim=1)
        decode_block3 = self.double_conv3(decode_block3)

        decode_block2 = self.conv_decode2(decode_block3)
        encode_block2 = self.att2(g=decode_block2, x=encode_block2)
        decode_block2 = torch.cat((encode_block2, decode_block2), dim=1)
        decode_block2 = self.double_conv2(decode_block2)

        decode_block1 = self.conv_decode1(decode_block2)
        encode_block1 = self.att1(g=decode_block1, x=encode_block1)
        decode_block1 = torch.cat((encode_block1, decode_block1), dim=1)
        decode_block1 = self.double_conv1(decode_block1)

        final_layer = self.final_layer(decode_block1)

        return torch.sigmoid(final_layer)