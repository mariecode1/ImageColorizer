import torch.nn as nn
import torch
from torchvision.models import vgg19
from utils import DiscriminatorLoss, PerceptualLoss

class UNetWithSkipConnections(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder = nn.ModuleList([self.conv_block(in_channels, 64)] + [self.conv_block(64 * 2 ** i, 64 * 2 ** (i + 1)) for i in range(4)])
        self.decoder = nn.ModuleList([self.upconv_block(64 * 2 ** (4 - i), 64 * 2 ** (3 - i)) for i in range(4)] + [self.upconv_block(128, out_channels, final=True)])

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def upconv_block(self, in_channels, out_channels, final=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() if final else nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU())
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        for i, layer in enumerate(self.decoder):
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = layer(x)
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.disc1 = self.build_discriminator(in_channels)
        self.disc2 = self.build_discriminator(in_channels)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def build_discriminator(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return [self.disc1(x), self.disc2(self.downsample(x))]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = nn.Sequential(*list(vgg19(pretrained=True).features[:16])).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        return nn.functional.l1_loss(gen_features, target_features)


import torch.nn as nn
import torch


class ColorizingNet(nn.Module):
    def __init__(self, device):
        super(ColorizingNet, self).__init__()
        self.generator = UNET()  # Ensure UNET is defined or imported
        self.discriminator = PatchDiscriminator()  # Ensure PatchDiscriminator is defined
        self.adversarial_loss = DiscriminatorLoss()
        self.pixelwise_loss = nn.L1Loss()

        self.register_buffer("l1_lambda", torch.tensor(100))  # Weight for L1 loss
        self.device = device
        self.to(device)

    def forward(self, x):
        """Forward pass through the generator."""
        return self.generator(x)  # Pass input through the generator



import torch.nn as nn
import torch

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNET, self).__init__()

        # Define encoder layers
        self.encoder = nn.ModuleList([
            self.down_block(in_channels, 64),
            self.down_block(64, 128),
            self.down_block(128, 256),
            self.down_block(256, 512)
        ])

        # Bottleneck
        self.bottleneck = self.down_block(512, 1024)

        # Define decoder layers
        self.decoder = nn.ModuleList([
            self.up_block(1024 + 512, 512),
            self.up_block(512 + 256, 256),
            self.up_block(256 + 128, 128),
            self.up_block(128 + 64, 64)
        ])

        # Final output layer
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            x = torch.cat((x, skips[-(i + 1)]), dim=1)  # Concatenate skip connection
            x = layer(x)

        x = self.final_layer(x)
        return x

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            # Resize skip connection to match decoder output
            skip = skips[-(i + 1)]
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                skip = F.interpolate(skip, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

            x = torch.cat((x, skip), dim=1)  # Concatenate skip connection
            x = layer(x)

        x = self.final_layer(x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)
