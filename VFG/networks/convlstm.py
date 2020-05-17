import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size=64, hidden_size=64, kernel_size, padding):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding)
        self.prev_state = None
    
    def init_hidden(self):
        self.prev_state = None

    def forward(self, input_, prev_state):
        print(input_.size())
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            self.prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = self.prev_state
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
       
        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell


class GeneratorF_convLSTM(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, c_dim, T, conv_dim=64, repeat_num=6):
        super(GeneratorF_static_ACR, self).__init__()
        self._name = 'generator_wasserstein_gan_f_convLSTM'
        self.T = T
        self.t = 0
        
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
        self.conv_lstm = ConvLSTMCell(64,64,3,1)
        
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
        

        self.fgmask_conv = nn.Conv2d(64,1,3,1,1,bias=False)
        self.satsig = nn.Sigmoid()
        self.reset_params()

    def reset_params(self):
        self.conv_lstm.init_hidden()
        self.t = 0
        
        
    def forward(self, If_prev_masked, OFprev2next): 
        x = torch.cat([If_prev_masked, OFprev2next], dim=1)
        features = self.main(x)
        hidden_ , cell_ = self.conv_lstm(features, self.features_)
        state = []
        color_mask = self.img_reg_packs(features_)
        att_mask = self.attention_reg_packs(features_)
        self.t = self.t + 1
        If_next_masked = att_mask * If_prev_masked + (1-att_mask)*color_mask
        fgmask = self.fgmask_conv(features_)
        fgmask = self.satsig(2.0*fgmask)
        If_next_masked = fgmask * If_next_masked + (1-fgmask) * -1.0 *torch.ones_like(If_next_masked).cuda()
        return  If_next_masked, fgmask
