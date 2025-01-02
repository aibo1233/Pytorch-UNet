""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .cbam import CBAM

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            DoubleConv(n_channels, 64),
            CBAM(64)  # 添加 CBAM 模块
        )
        self.down1 = nn.Sequential(
            Down(64, 128),
            CBAM(128)  # 添加 CBAM 模块
        )
        self.down2 = nn.Sequential(
            Down(128, 256),
            CBAM(256)  # 添加 CBAM 模块
        )
        self.down3 = nn.Sequential(
            Down(256, 512),
            CBAM(512)  # 添加 CBAM 模块
        )
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(
            Down(512, 1024 // factor),
            CBAM(1024 // factor)  # 添加 CBAM 模块
        )

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cbam1 = CBAM(512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.cbam2 = CBAM(256 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.cbam3 = CBAM(128)
        self.up4 = Up(128, 64, bilinear)
        self.cbam4 = CBAM(64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.cbam1(x)
        x = self.up2(x, x3)
        x = self.cbam2(x)
        x = self.up3(x, x2)
        x = self.cbam3(x)
        x = self.up4(x, x1)
        x = self.cbam4(x)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)