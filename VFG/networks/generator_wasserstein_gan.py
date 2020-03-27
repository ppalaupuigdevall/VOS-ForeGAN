import torch.nn as nn
import numpy as np
from networks import NetworkBase, NetworksFactory
import torch

class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, T=3):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'
        self.T = T
        self.t = 0
        self.factor = 2
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        def hook_function(m,i,o):
            self.lafeat = o
            
        self.main[-1].register_forward_hook(hook_function)

        # Prepare temporal skip-connections
        last_layer_dim = curr_dim
        # 0,1, .. T-1
        self.img_reg_packs = []
        self.attention_reg_packs = []
        self.reductor = [] 
        
        for i in range(T):
            layers_img = []
            layers_reductor = []
            layers_att = []

            layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
            layers_reductor.append(nn.ReLU(inplace=True))
            self.reductor.append(nn.Sequential(*layers_reductor))

            layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
            layers_img.append(nn.Tanh())
            self.img_reg_packs.append(nn.Sequential(*layers_img))

            layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
            layers_att.append(nn.Sigmoid())
            self.attention_reg_packs.append(nn.Sequential(*layers_att))

            last_layer_dim = int(curr_dim + last_layer_dim/self.factor)


    def reset_params(self):
        self.last_feature = torch.tensor([])
        
    def forward(self, x): 
        features = self.main(x)
        to_be_reduced = self.lafeat
        features = torch.cat([self.last_feature, features], dim=1) # Concat in channel dimension
        color_mask = self.img_reg_packs[self.t](features)
        att_mask = self.attention_reg_packs[self.t](features)
        self.last_feature = self.reductor[self.t](feat_map)
        if(self.t == self.T-1):
            self.reset_params()
        self.t = (self.t + 1)%self.T
        
        return self.img_reg(features), self.attetion_reg(features)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


if __name__ == '__main__':
    G =  NetworksFactory.get_by_name('generator_wasserstein_gan')
    print(G.main[-1])
