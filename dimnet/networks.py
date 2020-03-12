import numpy as np
from torch import no_grad
import torch.nn as nn
import torch.nn.functional as F


class VGGVox(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes=1251):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(1, 1)),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels=embed_dim, kernel_size=(27, 1), stride=(1, 1))
        )

        self.dense = nn.Sequential(
            nn.Linear(embed_dim, 512, bias=True),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=True)
        )

    def forward(self, x, embedding=False):
        x = x[:, None, :, :]
        x = self.model(x)
        x = F.avg_pool2d(x, (1, x.size()[3]), stride=1)
        x = x.view(x.size()[0], -1)
        if embedding:
            return x
        return self.dense(x)


class VGGVoxWrapper(VGGVox):
    def __init__(self, input_dim, embed_dim, num_classes=1251):
        super().__init__(input_dim, embed_dim, num_classes=1251)

    def forward(self, x, loss=False):
        if loss:
            # with no_grad():
            return self.dense(x)

        x = x[:, None, :, :]
        x = self.model(x)
        x = F.avg_pool2d(x, (1, x.size()[3]), stride=1)
        x = x.view(x.size()[0], -1)
        return x


class Dictionary(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1024, output_dim, bias=False)

    def forward(self, x):
        return self.linear(self.relu(self.linear1(x)))
        # return self.linear(x)


class CommonMLP(nn.Module):
    """
    MLP classifier that takes an input from the joint embedding space between faces and voices and classifies the
    ID of the person it belongs to.
    Uses Dropout after every odd numbered hidden layer.
    """

    def __init__(self, embed_dim, hidden_dims, num_classes, bias=True, dropout=0):
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]

        self.linear_in = nn.Linear(embed_dim, hidden_dims[0], bias=bias)
        self.dropout_in = nn.Dropout(p=dropout)
        self.relu_in = nn.ReLU(inplace=True)

        hidden_layers = []
        if hidden_dims > 1:
            for i in range(1, len(hidden_dims)):
                hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=bias))
                if i % 2 == 0:
                    hidden_layers.append(nn.Dropout(p=dropout))
                hidden_layers.append(nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.linear_out = nn.Linear(hidden_dims[-1], num_classes, bias=bias)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropout_in(x)
        x = self.relu_in(x)
        x = self.hidden_layers(x)
        x = self.linear_out(x)
        return x


class Resnet(nn.Module):
    def __init__(self, in_channels, img_size, block_channels, layer_blocks, kernel_sizes, strides, pool_size,
                 num_classes):
        """
        Resnet Module without expansion or bottleneck elements
        :param in_channels: the number of channels in the input data
        :param block_channels: the number of channels in each layer
        :param layer_blocks: the number of consecutive blocks in each layer
        :param strides: stride at the end of each layer
        :param num_classes:
        :param feat_dim:
        """
        super(Resnet, self).__init__()
        assert len(block_channels) == len(layer_blocks), \
            f"# block channels {block_channels} needs to equal # layer_blocks {layer_blocks}."

        # Initial layers
        self.layers = []
        conv1 = nn.Conv2d(in_channels, block_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.layers.append(conv1)
        self.layers.append(nn.BatchNorm2d(block_channels[0]))
        self.layers.append(nn.ReLU(inplace=True))

        # Residual block layers
        for i in range(len(block_channels)):
            num_blocks = layer_blocks[i]
            in_block_channels = block_channels[i] if i == 0 else block_channels[i - 1]
            block_layer = self._block_layer(in_block_channels, block_channels[i], kernel_sizes[i], strides[i],
                                            num_blocks)
            self.layers.append(block_layer)

        self.net = nn.Sequential(*self.layers)

        # pooling layer
        self.avg_pool = nn.AvgPool2d(pool_size)

        # linear output layer
        pooled_feature_map_size = (img_size // np.product(strides) // pool_size) ** 2
        self.linear_label = nn.Linear(block_channels[-1] * pooled_feature_map_size, num_classes, bias=False)

    def forward(self, x):
        embedding = self.net(x)

        output = self.avg_pool(embedding)
        output = output.reshape(output.shape[0], -1)

        label_output = self.linear_label(output)

        return output, label_output

    def _block_layer(self, in_channels, block_channels, kernel_size, stride, num_blocks):
        assert num_blocks >= 2, f"At least 2 blocks per layer required; {num_blocks} given."

        block_layer = []
        # first block
        block_layer.append(
            BasicBlock(in_channels, block_channels, kernel_size, stride=1)
        )
        # intermediate blocks
        for _ in range(num_blocks - 2):
            block_layer.append(
                BasicBlock(block_channels, block_channels, kernel_size, stride=1)
            )
        # downsample if necessary by striding
        block_layer.append(
            BasicBlock(block_channels, block_channels, kernel_size, stride=stride)
        )

        return nn.Sequential(*block_layer)


class BasicBlock(nn.Module):
    """
    Basic convolutional residual block with a skip connection
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        padding = int(kernel_size // 2)  # preserve image size
        self.reshape = stride > 1 or in_channels != out_channels  # whether x needs to be reshaped before adding

        self.straight = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if self.reshape:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.straight(x) + self.shortcut(x)  # add residual
        out = self.relu(out)
        return out