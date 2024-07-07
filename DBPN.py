import torch
import torch.nn as nn

# Function to initialize model weights with kaiming weights
def initialize_kaiming_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

# Defining the convolution block containing single convolution layer
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        # Initialize batchnormalization layer depending on the input parameter
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)

        self.activation = activation
        # Assign activation function layer depending on the input parameter
        if self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

# Defining the deconvolution block containing single deconvolution layer
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        # Initialize batchnormalization layer depending on the input parameter
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)

        self.activation = activation
        # Assign activation function layer depending on the input parameter
        if self.activation == 'prelu':
            self.act = torch.nn.PReLU()
            
    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
        
# Implementing the Up-Projection unit of the DBPN network
class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0
    
# Implementing the Dense Up-Projection unit of the DBPN network
class DenseUpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(DenseUpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0
    
# Implementing the Down-Projection unit of the DBPN network
class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

    
# Implementing the Dense Down-Projection unit of the DBPN network
class DenseDownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(DenseDownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


# Implementing the generator network of DBPN
class DBPN(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(DBPN, self).__init__()
          
        kernel = 8
        stride = 4
        padding = 2
        
        # Implementing the initial convolution layers
        # 3x3 conv layer
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        # 1x1 conv layer
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # Implementing the back-projection layers by alternating between up-projection and down-projection layers
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DenseDownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = DenseUpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = DenseDownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = DenseUpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = DenseDownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = DenseUpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = DenseDownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = DenseUpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = DenseDownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = DenseUpBlock(base_filter, kernel, stride, padding, 6)
        # Reconstruction the image output - 3x3 conv
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        # Initialize model with kaiming weights
        self.apply(initialize_kaiming_weights)
            
    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)

        h2 = self.up2(l1)
        l2 = self.down2(torch.cat((h2, h1), 1))

        h3 = self.up3(torch.cat((l2, l1), 1))
        l3 = self.down3(torch.cat((h3, h2, h1), 1))

        h4 = self.up4(torch.cat((l3, l2, l1), 1))
        l4 = self.down4(torch.cat((h4, h3, h2, h1), 1))

        h5 = self.up5(torch.cat((l4, l3, l2, l1), 1))
        l5 = self.down5(torch.cat((h5, h4, h3, h2, h1), 1))

        h6 = self.up6(torch.cat((l5, l4, l3, l2, l1), 1))
        l6 = self.down6(torch.cat((h6, h5, h4, h3, h2, h1), 1))

        h7 = self.up7(torch.cat((l6, l5, l4, l3, l2, l1), 1))
        x = self.output_conv(torch.cat((h7, h6, h5, h4, h3, h2, h1), 1))

        return x
    
# Implementing DenseBlock unit for discriminator contianing of linear layer
class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.activation = activation
        # Assign activation function layer depending on the input parameter
        if self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

# Implementing the discriminator network of DBPN
class Discriminator(nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu', norm ='batch'),
        )

        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu'),
            DenseBlock(base_filter * 16, 1, activation='sigmoid')
        )
        
        # Initialize model with kaiming weights
        self.apply(initialize_kaiming_weights)

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.flatten(start_dim=1)
        out = self.dense_layers(out)
        return out