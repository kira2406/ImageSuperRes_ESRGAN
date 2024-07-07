# defining n Efficient Sub-Pixel Convolutional Neural Network
import torch
import torch.nn as nn

class ESPCNN(nn.Module):
    
    # ESPCN uses 3 convolution layers with tanh as activation function along with an additional pixel shuffle layer
    def __init__(self, scale_factor, num_channels=1):
        super(ESPCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x