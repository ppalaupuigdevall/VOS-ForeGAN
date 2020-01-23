import torch
from torch import nn
from torch.autograd.variable import Variable


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self._linearlab = nn.Sequential(
            nn.Linear(10, 32 * 32)
        )

        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # x size: (BS, 1, 32, 32)
        # c size: (BS, 1)

        # Embed labels
        y = self.label_emb(c)
        # y size: (BS, 10)
        y = self._linearlab(y)
        # y size: (BS, 32*32)

        y = y.view(y.shape[0], 1, 32, 32)
        # y size: (BS, 1, 32, 32)

        # Concatenate
        x = torch.cat([x, y], 1)
        # x size: (BS, 2, 32, 32)

        # Forward data and labels
        x = self._conv1(x)

        x = self._conv2(x)
        x = self._conv3(x)
        x = x.view(-1, 512 * 4 * 4)
        # x size: (BS, 8192)

        x = self._fc(x)
        # x size: (BS, 1)

        return x
