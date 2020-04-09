import torch.nn as nn
import numpy as np
from networks.networks import NetworkBase, NetworksFactory
import torch

class GeneratorF(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF, self).__init__()
        self._name = 'generator_wasserstein_gan_f'
        self.T = T
        self.t = 0
        self.factor = 2
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
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
        self.lafeat = None
        def hook_function(m,i,o):
            self.lafeat = o
            
        self.main[-1].register_forward_hook(hook_function)

        # Prepare temporal skip-connections
        last_layer_dim = curr_dim
        # 0,1, .. T-1
        self.img_reg_packs = []
        self.attention_reg_packs = []
        self.reductor = [] 
        
        for i in range(T-1):
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
        
        self.img_reg_packs = nn.ModuleList(self.img_reg_packs)
        self.attention_reg_packs = nn.ModuleList(self.attention_reg_packs)
        self.reductor = nn.ModuleList(self.reductor)

        self.reset_params()

    def reset_params(self):
        self.last_features = [0] * self.T
        self.last_features[0] = torch.tensor([]).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next, If_next_warped): 
        
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)

        features = self.main(x)
        features_ = torch.cat([self.last_features[self.t], features], dim=1) # Concat in channel dimension

        print("Last feature = ", self.last_features[self.t].size())
        print("features_    = ", features_.size())
        color_mask = self.img_reg_packs[self.t](features_)
        att_mask = self.attention_reg_packs[self.t](features_)
        if(self.t<self.T-2):
            self.last_features[self.t+1] = self.reductor[self.t](features_)
        
        self.t = self.t + 1
        
        If_next_masked = att_mask * (If_next_warped + color_mask)  + ((1 - att_mask) * (If_next_warped + color_mask) -1 )
        return  If_next_masked, att_mask


class GeneratorB(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorB, self).__init__()
        self._name = 'generator_wasserstein_gan_b'
        self.T = T
        self.t = 0
        self.factor = 2
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
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
        self.lafeat = None
        def hook_function(m,i,o):
            self.lafeat = o
            
        self.main[-1].register_forward_hook(hook_function)

        # Prepare temporal skip-connections
        last_layer_dim = curr_dim
        # 0,1, .. T-1
        self.img_reg_packs = []
        self.attention_reg_packs = []
        self.reductor = [] 
        
        for i in range(T-1):
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

        self.img_reg_packs = nn.ModuleList(self.img_reg_packs)
        self.attention_reg_packs = nn.ModuleList(self.attention_reg_packs)
        self.reductor = nn.ModuleList(self.reductor)

        
        self.reset_params()

    def reset_params(self):
        self.last_features = [0] * self.T
        self.last_features[0] = torch.tensor([]).cuda()
        self.t = 0
    def forward(self, Ib_prev_masked, OFprev2next): 

        x = torch.cat([Ib_prev_masked, OFprev2next], dim=1)

        features = self.main(x)
        to_be_reduced = self.lafeat
        features = torch.cat([self.last_features[self.t], features], dim=1) # Concat in channel dimension
        color_mask = self.img_reg_packs[self.t](features)
        att_mask = self.attention_reg_packs[self.t](features)
        if(self.t < self.T -2):
            self.last_features[self.t+1] = self.reductor[self.t](features)
        self.t = self.t + 1
        # print("t = ", self.t, " T = ", self.T)
        # print("t = ", self.t)
        Ib_next = att_mask * (Ib_prev_masked) + (1-att_mask)*color_mask # Whole background 

        return Ib_next


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
