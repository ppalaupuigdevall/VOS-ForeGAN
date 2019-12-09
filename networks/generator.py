import torch
from torch import nn


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        #  Linear Layer - 100 for z (noise) and 10 for c (class labels)
        self._fc = torch.nn.Linear(100 + 10, 32 * 32 * 4 * 4)

        #  Sequential Container
        self._conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self._conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self._conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        # z size: (BS, 100)
        # c size: (BS)

        y = self.label_emb(c)
        # y size: (BS, 10)

        # Concat to condition noise with labels
        x = torch.cat([z, y], 1)
        # x size: (BS, 110)

        x = self._fc(x)

        x = x.view(x.shape[0], 1024, 4, 4)

        x = self._conv1(x)

        x = self._conv2(x)

        x = self._conv3(x)
        # x size: (BS, 1, 32, 32)

        return x

