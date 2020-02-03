import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VoiceBlock(nn.Module):
    def __init__(self, planes):
        super(VoiceBlock, self).__init__()
        self.conv1 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class SpeakerID(BaseModel):
    def __init__(self, input_dim, channel_dim, embed_dim,
                 kernel_size=3, stride=2, padding=1, alpha=1.5, num_classes=1251):
        super().__init__()
        channel_dims = [int(alpha**i*channel_dim) for i in range(4)]
        self.model = nn.Sequential(
             nn.Conv1d(input_dim, channel_dims[0], kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(channel_dims[0], affine=True),
             nn.ReLU(inplace=True),
             VoiceBlock(channel_dims[0]),
             nn.Conv1d(channel_dims[0], channel_dims[1], kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(channel_dims[1], affine=True),
             nn.ReLU(inplace=True),
             VoiceBlock(channel_dims[1]),
             nn.Conv1d(channel_dims[1], channel_dims[2], kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(channel_dims[2], affine=True),
             nn.ReLU(inplace=True),
             VoiceBlock(channel_dims[2]),
             nn.Conv1d(channel_dims[2], channel_dims[3], kernel_size, stride, padding, bias=False),
             nn.BatchNorm1d(channel_dims[3], affine=True),
             nn.ReLU(inplace=True),
             VoiceBlock(channel_dims[3]),
             nn.Conv1d(channel_dims[3], embed_dim, kernel_size, stride, padding, bias=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=True)
        )
    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1)
        return self.dense(x)

