import torch
import torch.nn as nn

# fsrcnn uses PReLU as activation function - implementhing FSRCNN(56, 12, 4) version
class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        # to maintain same size image for mid part, using padding 2 : conv(5,56,1) is defined below for first part
        # but as we are using 3 channels input image we are using conv(5,56,3)
        # decreasing number of output channels to faster procedssing
        self.firstpartconv = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.prelu56 = nn.PReLU(32)
            
        # midpart shrinking - conv(1,12,56)
        self.mpsconv = nn.Conv2d(32, 12, stride=1, kernel_size=1)
        self.prelu12 = nn.PReLU(12)

        # midpart non-linear 4* conv(3,12,12) with PReLU 12 activation function 
        # changing to get faster processing to 1 *  conv(3,12,12) with PReLU 12 activation function 
        self.mpsnlmapp = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
        )

        # midpart expanding - conv(1,56,12)
        self.mpeconv = nn.Conv2d(12, 32, stride=1, kernel_size=1)

        # lastpart - deconv(9,3,56)
        self.deconvolution = nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):
        # first part = feature extraction
        x = self.firstpartconv(x)
        x = self.prelu56(x)

        # mid part shrinking
        x = self.mpsconv(x)
        x = self.prelu12(x)

        #mid part non-linear mapping
        x = self.mpsnlmapp(x)

        # mid part expanding
        x = self.mpeconv(x)
        x = self.prelu56(x)

        # last part - deconvolution 
        x = self.deconvolution(x)
        return x

