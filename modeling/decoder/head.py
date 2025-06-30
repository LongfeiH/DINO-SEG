import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SE_SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=151, inter_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEModule(inter_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(inter_channels, num_classes, kernel_size=1)

    def forward(self, x, images):
        H, W = images.shape[-2:]
        x = self.conv1(x)
        x = self.se(x)
        x = self.dropout(x)
        x = self.classifier(x)
        out = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class CBAM_SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=151, inter_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(inter_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(inter_channels, num_classes, kernel_size=1)

    def forward(self, x, images):
        H, W = images.shape[-2:]
        x = self.conv1(x)
        x = self.cbam(x)
        x = self.dropout(x)
        x = self.classifier(x)
        out = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return out
