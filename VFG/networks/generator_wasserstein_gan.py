import torch.nn as nn
import numpy as np
from networks.networks import NetworkBase, NetworksFactory
import torch
import torch.nn.functional as F

###### FOREGROUND #######

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
        self.img_reg_packs = None
        self.attention_reg_packs = None
        self.reductor = None
        
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
    
        self.img_reg_packs = nn.Sequential(self.img_reg_packs)
        self.attention_reg_packs = nn.Sequential(self.attention_reg_packs)
        self.reductor = nn.Sequential(self.reductor)
        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.last_features = torch.tensor([]).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next): 
        
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        # x = If_next_warped
        features = self.main(x)
        features_ = torch.cat([self.last_features[self.t], features], dim=1) # Concat in channel dimension

        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1
        
        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask # experiment_4
        fgmask = self.fgmask_conv(features)
        fgmask = self.satsig(20*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * If_next_masked
        return  If_next_masked, fgmask


class GeneratorF_static_ACR(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
       
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features_)
        fgmask = self.satsig(2.0*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        return  If_next_masked, fgmask


class GeneratorF_static_ACR_v1(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_v1, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_v1'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features)
        fgmask = self.satsig(30.0*fgmask)
        # If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        If_next_masked = fgmask*If_next_masked + (1-fgmask)*If_next_masked
        return  If_next_masked, fgmask

class GeneratorF_static_ACR_v1_dark(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_v1_dark, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_v1_dark'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features)
        fgmask = self.satsig(4.0*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        # If_next_masked = fgmask*If_next_masked + (1-fgmask)*If_next_masked
        return  If_next_masked, fgmask

class TransformEstimator(nn.Module):
    def __init__(self):
        super(TransformEstimator, self).__init__()

        estimate_transform_layers = []
        estimate_transform_layers.append(nn.Conv2d(64, 32, 3, 1, 1)) # 64x224x416
        estimate_transform_layers.append(nn.MaxPool2d(2,2)) # 32x112x208
        estimate_transform_layers.append(nn.ReLU(True))
        estimate_transform_layers.append(nn.Conv2d(32,16,3,1,1)) # 16x56x104
        estimate_transform_layers.append(nn.MaxPool2d(2,2))
        estimate_transform_layers.append(nn.ReLU(True))
        estimate_transform_layers.append(nn.MaxPool2d(2,2))
        estimate_transform_layers.append(nn.Conv2d(16,8,3,1,1)) # 8x28x52
        estimate_transform_layers.append(nn.ReLU(True))
        estimate_transform_layers.append(nn.Conv2d(8,1,3,1,1))
        estimate_transform_layers.append(nn.ReLU(True))
        self.affine_tr_estimator_convs = nn.Sequential(*estimate_transform_layers)
        estimate_transform_layers_linears = []
        estimate_transform_layers_linears.append(nn.Linear(28*52, 3000))
        estimate_transform_layers_linears.append(nn.ReLU(True))
        estimate_transform_layers_linears.append(nn.Linear(3000, 6))
        self.affine_tr_estimator_linears = nn.Sequential(*estimate_transform_layers_linears)

    def forward(self, x):
        x = self.affine_tr_estimator_convs(x)
        x = self.affine_tr_estimator_linears(x.view(-1, ))
        return x

class GeneratorF_static_ACR_v1_dark_m(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_v1_dark_m, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_v1_dark_m'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.affine_tr_estimator = TransformEstimator()


        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next,maska): 
        x = torch.cat([If_prev_masked, OFprev2next,maska], dim=1)
        features = self.main(x)
        
        theta = self.affine_tr_estimator(features)
        grid = F.affine_grid(theta.view(1,2,3), features.size())
        features = F.grid_sample(features, grid)

        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features)
        fgmask = self.satsig(20.0*fgmask) # v10_Bf i v10 amb 4 ||| v10_satsig_20 amb 20 i laltre tambe
        # If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda() # Aixo eren els forestgan_rnn_v10 i forestgan_rnn_v10_BFi v10_satsig_20_dark
        If_next_masked = fgmask*If_next_masked + (1-fgmask)*If_next_masked # aquest es el satsig_20_basliene
        return  If_next_masked, fgmask

class GeneratorF_static_ACR_v1_antic(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T,use_moment, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_v1_antic, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_v1_antic'
        self.T = T
        self.t = 0
        self.factor = 1
        self.use_moments = use_moment
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features)
        fgmask = self.satsig(30.0*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * If_next_masked
        if self.use_moments:
            return  If_next_masked, fgmask, features
        else:
            return  If_next_masked, fgmask


class GeneratorF_static_ACR_mask_from_fg(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_mask_from_fg, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_mask_from_fg'
        self.T = T
        self.t = 0
        self.factor = 1
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

        # Prepare temporal skip-connections
        last_layer_dim = curr_dim
        # 0,1, .. T-1
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(3,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(If_next_masked)
        fgmask = self.satsig(2.0*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        return  If_next_masked, fgmask



class GeneratorF_static_ACR_noOF(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR_noOF, self).__init__()
        self._name = 'generator_wasserstein_gan_f_static_ACR_noOF'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0
        
    def forward(self, If_prev_masked, Inext): 
        # Inext_flipped = Inext.flip(3)
        x = torch.cat([If_prev_masked, Inext], dim=1)
        features = self.main(x)
        features_ = self.last_features + features
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        if(self.t<self.T-1):
            self.last_features = self.reductor(features_)
        self.t = self.t + 1

        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask 
        fgmask = self.fgmask_conv(features_)
        fgmask = self.satsig(10.0*fgmask)
        
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        
        return  If_next_masked, fgmask


####### BACKGROUND #######

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
            if(i==0):
                self.factor = self.factor + 1


        self.img_reg_packs = nn.ModuleList(self.img_reg_packs)
        self.attention_reg_packs = nn.ModuleList(self.attention_reg_packs)
        self.reductor = nn.ModuleList(self.reductor)

        
        self.reset_params() # commit

    def reset_params(self):
        self.last_features = [0] * self.T
        self.last_features[0] = torch.tensor([]).cuda()
        self.t = 0
    def forward(self, Ib_prev_masked): 

        # x = torch.cat([Ib_prev_masked, OFprev2next], dim=1)
        x = Ib_prev_masked
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
        Ib_next = att_mask * (Ib_prev_masked) + (1-att_mask)*color_mask # Whole background cuda0
        # Ib_next = att_mask * (color_mask) + (1-att_mask)*Ib_prev_masked  # Whole background cuda0
        

        return Ib_next


class GeneratorB_static_ACR(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorB_static_ACR, self).__init__()
        self._name = 'generator_wasserstein_gan_b_static_ACR'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0

    def forward(self, Ib_prev_masked): 
        x = Ib_prev_masked
        features = self.main(x)
        features = self.last_features + features
        color_mask = self.img_reg_packs(features)
        att_mask = self.attention_reg_packs(features)
        if(self.t < self.T -1):
            self.last_features = self.reductor(features)
        self.t = self.t + 1
        Ib_next = att_mask * (Ib_prev_masked) + (1-att_mask)*color_mask # Whole background cuda0
        return Ib_next

class GeneratorB_static_ACR_OF(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorB_static_ACR_OF, self).__init__()
        self._name = 'generator_wasserstein_gan_b_static_ACR'
        self.T = T
        self.t = 0
        self.factor = 1
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
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)
        layers_reductor = []
        layers_reductor.append(nn.Conv2d(last_layer_dim, int(last_layer_dim/self.factor), kernel_size=3, stride=1, padding=1, bias=False))
        layers_reductor.append(nn.Tanh())
        self.reductor = nn.Sequential(*layers_reductor) 

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.last_features = torch.zeros(64,224,416).cuda()
        self.t = 0

    def forward(self, Ib_prev_masked, OFprev2next): 
    
        x = torch.cat([Ib_prev_masked,OFprev2next], dim=1)
        features = self.main(x)
        features = self.last_features + features
        color_mask = self.img_reg_packs(features)
        att_mask = self.attention_reg_packs(features)
        if(self.t < self.T -2):
            self.last_features = self.reductor(features)
        self.t = self.t + 1
        Ib_next = att_mask * (Ib_prev_masked) + (1-att_mask)*color_mask # Whole background cuda0
        return Ib_next



class GeneratorB_static_ACR_single_img(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorB_static_ACR_single_img, self).__init__()
        self._name = 'generator_wasserstein_gan_b_static_ACR_single_image'
        self.T = T
        self.t = 0
        self.factor = 1
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
    
        
        # Prepare temporal skip-connections
        last_layer_dim = curr_dim
        # 0,1, .. T-1
        layers_img = []
        layers_img.append(nn.Conv2d(last_layer_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_img.append(nn.Tanh())
        self.img_reg_packs = nn.Sequential(*layers_img)
        layers_att = []
        layers_att.append(nn.Conv2d(last_layer_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers_att.append(nn.Sigmoid())
        self.attention_reg_packs = nn.Sequential(*layers_att)


    def forward(self, Ib_prev_masked): 
    
        x = Ib_prev_masked
        features = self.main(x)
        color_mask = self.img_reg_packs(features)
        att_mask = self.attention_reg_packs(features)
        Ib_next = att_mask * (Ib_prev_masked) + (1-att_mask)*color_mask # Whole background cuda0
        return Ib_next


##########################

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