import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(self, in_dim, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=s),
                nn.Conv2d(in_dim, in_dim // len(pool_sizes), kernel_size=1),
                nn.ReLU(inplace=True)
            ) for s in pool_sizes
        ])
        self.out = nn.Conv2d(in_dim + in_dim // len(pool_sizes) * len(pool_sizes), in_dim, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W]
        ppm_outs = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(out)
        out = torch.cat(ppm_outs, dim=1)
        return self.out(out)


class MultiLayerPPMFusion(nn.Module):
    def __init__(self, num_layers=4, in_dim=384):
        super().__init__()
        self.ppm = PPM(in_dim)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_layers * in_dim, in_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, h, w): 
        B = features[0].shape[0]
        fused = []
        for i, f in enumerate(features):  # f: [B, 1369, C]
            x = f.permute(0, 2, 1).reshape(B, -1, h, w)  # [B, C, H, W]
            x = self.ppm(x)  # [B, C, H, W]
            fused.append(x)
        out = torch.cat(fused, dim=1)  # [B, 12*C, H, W]
        return self.fusion_conv(out) 