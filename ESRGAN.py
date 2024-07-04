import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.leaky_relu(self.conv1(x))
        
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        
        out5 = self.conv5(torch.cat([x, out1, out2, out3, out4], 1))
        
        return out5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23, growth_channels=32):
        super(ESRGAN, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(num_features, growth_channels) for _ in range(num_blocks)])
        self.conv_second = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        out = self.conv_first(x)
        out = self.rrdb_blocks(out)
        out = self.conv_second(out) + out
        out = self.upsampling(out)
        out = self.conv_last(out)
        return out
  
