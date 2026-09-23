import torch
import torch.nn as nn


class ZPool(nn.Module):
    """Z-Pool: concatenate max-pooling and average-pooling along channel dimension."""
    def forward(self, x):
        # x: (B, C, H, W) -> (B, 2, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True)[0]
        return torch.cat([maxv, avg], dim=1)


class TBAM(nn.Module):
    """Three-Branch Attention Mechanism.

    Branch 1: rotate along height dimension and learn channel-height attention.
    Branch 2: rotate along width dimension and learn channel-width attention.
    Branch 3: spatial attention.
    The three attention maps are averaged and multiplied with the input.
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.zpool = ZPool()
        self.conv_h = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv_w = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv_s = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # Branch 1: height rotation
        x_h = x.permute(0, 2, 1, 3)          # (B, H, C, W)
        z_h = self.zpool(x_h)                # (B, 2, C, W)
        a_h = self.sigmoid(self.conv_h(z_h)) # (B, 1, C, W)
        a_h = a_h.permute(0, 2, 1, 3)        # (B, C, 1, W)
        a_h = a_h.expand(-1, -1, h, -1)      # (B, C, H, W)

        # Branch 2: width rotation
        x_w = x.permute(0, 3, 2, 1)          # (B, W, C, H)
        z_w = self.zpool(x_w)                # (B, 2, C, H)
        a_w = self.sigmoid(self.conv_w(z_w)) # (B, 1, C, H)
        a_w = a_w.permute(0, 2, 3, 1)        # (B, C, H, 1)
        a_w = a_w.expand(-1, -1, -1, w)      # (B, C, H, W)

        # Branch 3: spatial attention
        z_s = self.zpool(x)                  # (B, 2, H, W)
        a_s = self.sigmoid(self.conv_s(z_s)) # (B, 1, H, W)
        a_s = a_s.expand(-1, c, -1, -1)      # (B, C, H, W)

        att = (a_h + a_w + a_s) / 3.0
        return x * att
