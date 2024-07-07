#defining Super Resolution Convolutional Neural Network
import torch
import torch.nn as nn

# SRCNN uses 3 convolution layers with relu as activation function
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        
        super(SRCNN, self).__init__()

        # best setting kernel size mentioned in paper is filter1 = 9, filter2 = 5, filter3 = 5 
        # number of channels n1 = 64, n2 = 32 and n3 = number of channels of original image
        # formula for same image size after a[pplying kernel - padding should be (kernel-1)/2

        # checked with 11-7-7 version but as it was taking too much time, revert back to 9-5-5 version
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        # defininf relu acrtivation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2 non linear layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # linear mapping layer
        x = self.conv3(x)
        return x