from torch import no_grad
import torch.nn as nn
import torch.nn.functional as F


class VGGVox(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes=1251):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(7,7), stride=(2,2), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride=(2,2)),
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(2,2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride=(1,1)),
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),
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
        x = x[:,None,:,:]
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

        x = x[:,None,:,:]
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