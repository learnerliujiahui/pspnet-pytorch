# this file provide all kinds of module
import torch
import torch.nn as nn

class DSMModule(nn.Module):
    """only for example(JUL.29)"""
    def __init__(self, n_classes, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class twoBranchNet(nn.Module):
    """call the PSPNet Module and DSM module, and merge two branch"""
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass