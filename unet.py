import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(LightningModule):

    def __init__(self, n_channels, n_classes, bilinear=True, div=2):

        super(UNet, self).__init__()

        self.name = 'UNET'

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64//div)
        self.down1 = Down(64//div, 128//div)
        self.down2 = Down(128//div, 256//div)
        self.down3 = Down(256//div, 512//div)
        factor = 2 if bilinear else 1
        self.down4 = Down(512//div, 1024//div // factor)
        self.up1 = Up(1024//div, 512//div // factor, bilinear)
        self.up2 = Up(512//div, 256//div // factor, bilinear)
        self.up3 = Up(256//div, 128//div // factor, bilinear)
        self.up4 = Up(128//div, 64//div, bilinear)
        self.outc = OutConv(64//div, n_classes)

    def forward(self, x, f):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
    
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits.squeeze(1), x

    def training_step(self, train_batch, batch_idx):

        series_input, features, _, series_target = train_batch

        series_output = self.forward(series_input, features)[0]

        #loss = torch.mean(torch.square(series_target - series_output))
        loss = F.mse_loss(series_target, series_output, reduction='sum') 

        with torch.no_grad():
            train_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('train_loss', loss.cpu().item())
        self.log('train_mae', train_mae)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series_input, features, _, series_target = val_batch

        series_output = self.forward(series_input, features)[0]

        #loss = torch.mean(torch.square(series_target - series_output))
        loss = F.mse_loss(series_target, series_output, reduction='sum') 

        #TODO: nograd needed?
        val_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_mae', val_mae)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/5, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_mae'}

