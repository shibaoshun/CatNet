
import torch.nn as nn
import torch
import torch.nn.functional as F


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class single_conv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x),self.conv[0].weight.data

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.than=nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        # x=self.than(x)
        return x


class globalFeature(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, inSize, outSize):
        super(globalFeature, self).__init__()
        self.global_feature = nn.Sequential(
            nn.Linear(inSize, outSize),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.global_feature_1 = nn.Sequential(
            nn.Linear(outSize, outSize),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, y2, x):
        y = torch.mean(x, dim=(2, 3))
        y1 = self.global_feature(y)

        y = self.global_feature_1(y1)
        y1 = torch.unsqueeze(y1, dim=2)
        y1 = torch.unsqueeze(y1, dim=3)

        y = torch.unsqueeze(y, dim=2)
        y = torch.unsqueeze(y, dim=3)

        glob = y2 * y1 + y
        # glob=torch.cat((glob,size),dim=1)
        return glob

class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


class FFC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FFC, self).__init__()
        self.fr_net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_ch, momentum=0.9, eps=1e-04, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.fr_net1 = nn.Sequential(
            nn.Conv2d(2*in_ch, 2*out_ch, 1, padding=0, bias=True),
            nn.BatchNorm2d(2*out_ch, momentum=0.9, eps=1e-04, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*in_ch, 2*out_ch, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.CALayer = CALayer(in_ch)
        self.CALayer1 = CALayer(2*in_ch)



    def forward(self, convglo):
        aa=convglo.size(1)

        FR_net1 = self.fr_net(convglo)
        m = torch.fft.fft2(convglo)

        m1 = torch.real(m)
        m2 = torch.imag(m)
        m3 = torch.cat([m1, m2], dim=1)
        me = self.fr_net1(m3)  #

        me = self.CALayer1(me)

        a = me[:, :aa, :, :]
        b = me[:, aa:2*aa, :, :]
        m4 = a + b * 1j
        mee = torch.fft.ifft2(m4)
        mee = torch.abs(mee)
        out = FR_net1 + convglo + mee

        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 32),
            single_conv(32, 32),
            single_conv(32, 32)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64),
            single_conv(64, 64),
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
        )

        self.down3 = nn.AvgPool2d(2)
        self.conv3 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
        )

        self.down4 = nn.AvgPool2d(2)
        self.conv4 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.glo = globalFeature(256, 256)
        self.convglo = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),

        )
        self.convglo1 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )
        self.glo1 = globalFeature(256, 256)

        self.up1 = up(256, 256)
        self.convup1 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.up2 = up(256, 128)
        self.convup2 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
        )

        self.up3 = up(128, 64)
        self.convup3 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )
        self.up4 = up(64, 32)
        self.convup4 = nn.Sequential(
            single_conv(32, 32),
            single_conv(32, 32)
        )
        self.fr_net = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0, bias=True),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-04, affine=True),
            nn.ReLU(inplace=True),
        )

        self.fr_net1 = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=0, bias=True),
            nn.BatchNorm2d(512, momentum=0.9, eps=1e-04, affine=True),
            nn.ReLU(inplace=True),
        )

        self.FFC = FFC(256, 256)

        self.FFC1 = FFC(128, 128)
        self.FFC2 = FFC(64, 64)
        self.FFC3 = FFC(32, 32)


        self.outc = outconv(32, 3)

    def forward(self, x,level):
        img = torch.cat((level, x), 1)

        inx = self.inc(img)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        down3 = self.down3(conv2)
        conv3 = self.conv3(down3)

        down4 = self.down4(conv3)
        conv4= self.conv4(down4)

        glo = self.glo(down4, conv4)
        convglo = self.convglo(glo)


        out = self.FFC(convglo)
        convglo = out


        convglo1 = self.convglo1(convglo)
        glo1 = self.glo1(convglo, convglo1)

        conv3 = self.FFC(conv3)
        up1 = self.up1(glo1, conv3)
        convup1 = self.convup1(up1)

        conv2 = self.FFC1(conv2)
        up2 = self.up2(convup1, conv2)
        convup2 = self.convup2(up2)

        conv1 = self.FFC2(conv1)
        up3 = self.up3(convup2, conv1)
        convup3 = self.convup3(up3)

        inx = self.FFC3(inx)
        up4 = self.up4(convup3, inx)
        convup4 = self.convup4(up4)

        out = self.outc(convup4)
        return out
