import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.GroupNorm(32, out_channels),   # Group normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.GroupNorm(32, out_channels),   # Group normalization
            nn.GELU(),   # GELU activation function
        )

        self.same_channels = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, hidden_size=256, num_classes=10, image_size=28):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes 
        self.image_size = image_size

        self.init_conv = ResidualConvBlock(in_channels, hidden_size, is_res=True)

        self.down1 = UnetDown(hidden_size, hidden_size)
        self.down2 = UnetDown(hidden_size, 2*hidden_size)

        # self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        self.time_embed_1 = EmbedFC(1, 2*hidden_size)
        self.time_embed_2 = EmbedFC(1, hidden_size)

        self.context_embed_1 = EmbedFC(num_classes, 2*hidden_size)
        self.context_embed_2 = EmbedFC(num_classes, hidden_size)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_size, 2*hidden_size, self.image_size//4, self.image_size//4),
            nn.GroupNorm(8, 2 * hidden_size),
            nn.GELU()
        )

        self.up1 = UnetUp(4 * hidden_size, hidden_size)
        self.up2 = UnetUp(2 * hidden_size, hidden_size)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, hidden_size), # normalize
            nn.ReLU(),
            nn.Conv2d(hidden_size, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)  
        down1 = self.down1(x)
        down2 = self.down2(down1)

        to_vec = nn.Sequential(
            nn.AvgPool2d(down2.shape[2]),
            nn.GELU(),
        )
        hiddenvec = to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.num_classes).to(x.device)

        cemb1 = self.context_embed_1(c).view(-1, 2*self.hidden_size, 1, 1)
        tembed1 = self.time_embed_1(t).view(-1, 2*self.hidden_size, 1, 1)
        cemb2 = self.context_embed_2(c).view(-1, self.hidden_size, 1, 1)
        tembed2 = self.time_embed_2(t).view(-1, self.hidden_size, 1, 1)


        up0 = self.up0(hiddenvec) 
        up1 = self.up1(cemb1 * up0 + tembed1, down2)
        up2 = self.up2(cemb2 * up1 + tembed2, down1)
        out = self.out(torch.cat((up2, x), 1))
        return out


# unconditional version
class UNet2DModel(nn.Module):
    def __init__(self, in_channels, down_block_channels, image_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size 
        self.down_block_channels = down_block_channels
        self.num_downblocks = len(down_block_channels) 

        self.init_conv = ResidualConvBlock(in_channels, down_block_channels[0], is_res=True)

        down_blocks = []
        in_feats = self.down_block_channels[0]
        for i in range(self.num_downblocks):
            out_channels = self.down_block_channels[i]
            downlayer = UnetDown(in_feats, out_channels)
            down_blocks.append(downlayer)
            in_feats = out_channels

        self.down_blocks = nn.ModuleList(down_blocks)

        # self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        # print('down_block_channels:', self.down_block_channels)
        time_embeddings = []
        for channels in reversed(self.down_block_channels):
            temb = EmbedFC(1, channels)
            time_embeddings.append(temb)
        self.time_embeddings = nn.ModuleList(time_embeddings)
        # print('down_block_channels:', self.down_block_channels)

        down_stride = int(2**len(self.down_block_channels))
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(self.down_block_channels[-1], self.down_block_channels[-1], self.image_size//down_stride, self.image_size//down_stride),
            nn.GroupNorm(32, self.down_block_channels[-1]),
            nn.GELU(),
        )

        up_blocks = []
        
        self.down_block_channels.reverse()
        # print('down_block_channels:', self.down_block_channels)

        for i, channels in enumerate(self.down_block_channels[:-1]):
            in_channels = 2 * channels 
            out_channels  = self.down_block_channels[i+1]
            uplayer = UnetUp(in_channels, out_channels)
            up_blocks.append(uplayer)

        uplayer = UnetUp(self.down_block_channels[-1]*2, self.down_block_channels[-1])
        up_blocks.append(uplayer)
        self.up_blocks = nn.ModuleList(up_blocks)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * self.down_block_channels[-1], self.down_block_channels[-1], 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(32, self.down_block_channels[-1]), # normalize
            nn.GELU(),
            nn.Conv2d(self.down_block_channels[-1], self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t):
        init_conv_out = self.init_conv(x)  

        down_layer_outputs = [None for _ in range(len(self.down_blocks))]
        x = init_conv_out
        for i in range(len(self.down_blocks)):
            down_layer_outputs[i] = self.down_blocks[i](x) 
            x = down_layer_outputs[i] 
            # print('down:', 'i:', i, 'x:', x.shape)
        
        to_vec = nn.Sequential(
            nn.AvgPool2d(x.shape[2]),
            nn.GELU(),
        )

        hiddenvec = to_vec(x)
        # print('to_vec:', hiddenvec.shape)

        x = self.up0(hiddenvec) 
        down_layer_outputs.reverse()
        for i, down_x in enumerate(down_layer_outputs):
            temb = self.time_embeddings[i](t).view(1, -1, 1, 1)
            # print('i:', i, 'temb:', temb.shape, 'x_prev:', x.shape)
            x = self.up_blocks[i](temb + x, down_x)

        out = self.out(torch.cat((x, init_conv_out), 1))

        # tembed1 = self.time_embed_1(t).view(-1, 2*self.hidden_size, 1, 1)
        # tembed2 = self.time_embed_2(t).view(-1, self.hidden_size, 1, 1)

        # up1 = self.up1(cemb1 * up0 + tembed1, down2)
        # up2 = self.up2(cemb2 * up1 + tembed2, down1)

        # out = self.out(torch.cat((up2, x), 1))
        return out

# in_channels = 3 
# down_block_channels = [128, 128, 256, 256]
# image_size=128
# model = UNet2DModel(in_channels, down_block_channels, image_size).to(device)
# image = torch.randn(1,in_channels, image_size, image_size).to(device)
# t = torch.tensor(1.).to(device)
# out = model(image, t)
# print(out.shape)