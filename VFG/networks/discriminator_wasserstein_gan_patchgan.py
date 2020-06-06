import torch.nn as nn
import numpy as np
from networks.networks import NetworkBase, NetworksFactory
import torch
class Discriminator(NetworkBase):
    """Discriminator. PatchGAN. Processes patches independently
        Inputs are (B, Nx, Ny, 3, Hp, Wp) --> (B*Nx*Ny, 3, Hp, Wp)
        - B: bach size
        - Nx, Ny: number of patches in each dimension of the image
        - Hp, Wp: Height and width of each patch
    """
    def __init__(self, conv_dim=64, repeat_num=4):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_wasserstein_gan'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        
        self.conv1 = nn.Sequential(*[nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(2,4), stride=(1,1), padding=(0,0), bias=False), nn.LeakyReLU(0.01, inplace=True)])
        
        curr_dim = curr_dim * 2
        self.conv2 = nn.Sequential(*[nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(2,2), stride=(1,1), padding=(0,0), bias=False), nn.LeakyReLU(0.01, inplace=True)])
        curr_dim = curr_dim * 2
        self.conv3 = nn.Conv2d(curr_dim, 1, kernel_size=(1,3), padding=(0,0))
        

    def forward(self, x):
        h = self.main(x)
        h = self.conv1(h)
        h = self.conv2(h)
        out_real = self.conv3(h)
        return out_real

class Discriminator_G(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=224, conv_dim=64, repeat_num=6):
        super(Discriminator_G, self).__init__()
        self._name = 'discriminator_wgan_G'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1_fg = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_mask = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, is_fg):
        
        if is_fg:
            h = self.main(x)
            out_real = self.conv1_fg(h)
        else:
            x = x.expand(1,3,224,416)            
            h = self.main(x)
            out_real = self.conv1_mask(h)
        return out_real.squeeze()


class Discriminator_M(NetworkBase):
    """Discriminator. PatchGAN. Processes patches independently
        Inputs are (B, Nx, Ny, 3, Hp, Wp) --> (B*Nx*Ny, 3, Hp, Wp)
        - B: bach size
        - Nx, Ny: number of patches in each dimension of the image
        - Hp, Wp: Height and width of each patch
    """
    def __init__(self, conv_dim=64, repeat_num=4):
        super(Discriminator_M, self).__init__()
        self._name = 'discriminator_wasserstein_gan_M'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(*[nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(2,4), stride=(1,1), padding=(0,0), bias=False), nn.LeakyReLU(0.01, inplace=True)])
        
        curr_dim = curr_dim * 2
        self.conv2 = nn.Sequential(*[nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(2,2), stride=(1,1), padding=(0,0), bias=False), nn.LeakyReLU(0.01, inplace=True)])
        curr_dim = curr_dim * 2
        self.conv3 = nn.Conv2d(curr_dim, 1, kernel_size=(1,3), padding=(0,0))
        self.conv4 = nn.Conv2d(curr_dim,1, kernel_size=(1,3),padding=(0,0))

    def forward(self, x, is_fg=True):
        if(is_fg):
            # x is a fg
            h = self.main(x)
            h = self.conv1(h)
            h = self.conv2(h)
            out_real = self.conv3(h)
        else:
            x = x.expand(1,3,224,416)
            # is mask
            h = self.main(x)
            h = self.conv1(h)
            h = self.conv2(h)
            out_real = self.conv4(h)
        return out_real


if __name__ == '__main__':
    D = NetworksFactory.get_by_name('discriminator_wasserstein_gan')
    print(D)
    x = torch.rand(30,3,60,112)
    print(D(x).size())