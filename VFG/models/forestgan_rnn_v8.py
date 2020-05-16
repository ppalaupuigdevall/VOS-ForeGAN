import torch
from collections import OrderedDict
from torch.autograd import Variable
from models.models import BaseModel
from networks.networks import NetworksFactory
import os
import utils.util as util
import numpy as np
import torch.nn.functional as F
from data.dataset_davis import tensor2im
import cv2

class ForestGANRNN_v8(BaseModel):
    def __init__(self, opt):
        
        super(ForestGANRNN_v8, self).__init__(opt)
        self._name = 'forestgan_rnn_v8'
        self._opt = opt
        self._T = opt.T
        self._extra_ch_Gf = 2
      

        # create networks
        self._init_create_networks()
        self._is_train = opt.is_train
        
        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        # init
        self._init_losses()


    def set_input(self, sample):
        self._imgs = sample['imgs'] # [i_0, i_1, ..., i_t-1] REAL IMAGES
        self._OFs = sample['OFs'] # [of_0, of_1, ..., of_t-2]
        self._first_fg = sample['mask_f']
        self._real_mask = sample['mask']
        self._transformed_mask = sample['transformed_mask']
        self._move_inputs_to_gpu(0)
        

    def _move_inputs_to_gpu(self, t):
        if(t==0):
            self._visual_masks = []
            self._visual_fgs = []
        
            self._visual_fakes = []
            self._next_frame_imgs_ori = self._imgs[t+1].cuda()
            self._curr_OFs = self._OFs[t].cuda()
            self._curr_f = self._first_fg.cuda()
            self._first_fg = self._first_fg.cuda()
            self._real_mask = self._real_mask.cuda()
            self._transformed_mask = self._transformed_mask.cuda()
        else:
            self._curr_OFs = self._OFs[t].cuda()
            self._next_frame_imgs_ori = self._imgs[t+1].cuda()        


   

    def _init_create_networks(self):
        
        self._Gf = self._create_generator_f()
        self._Gf.init_weights()
        if len(self._gpu_ids) > 1:
            self._Gf = torch.nn.DataParallel(self._Gf, device_ids=self._gpu_ids)
        self._Gf.cuda()

        self._Df = self._create_discriminator_f()
        self._Df.init_weights()
        if len(self._gpu_ids) > 1:
            self._Df = torch.nn.DataParallel(self._Df, device_ids=self._gpu_ids)
        self._Df.cuda()

    

    def _create_generator_f(self):
        return NetworksFactory.get_by_name('generator_wasserstein_gan_f_static_ACR', c_dim=self._extra_ch_Gf, T=self._opt.T)

    
    def _create_discriminator_f(self):
        return NetworksFactory.get_by_name('discriminator_wasserstein_gan_M')
    
  
    def _init_train_vars(self):
        self._current_lr_Gf = self._opt.lr_Gf
        self._current_lr_Df = self._opt.lr_Df

        # initialize optimizers
        self._optimizer_Gf = torch.optim.Adam(self._Gf.parameters(), lr=self._current_lr_Gf,
                                             betas=[self._opt.Gf_adam_b1, self._opt.Gf_adam_b2])
        self._optimizer_Df = torch.optim.Adam(self._Df.parameters(), lr=self._current_lr_Df,
                                             betas=[self._opt.Df_adam_b1, self._opt.Df_adam_b2])
        
        
    def _init_losses(self):
        # define loss functions
        self._criterion_Gs_rec = torch.nn.L1Loss().cuda()

        # init losses G
        self._loss_g_fg = torch.cuda.FloatTensor([0])
        self._loss_g_fb_rec = torch.cuda.FloatTensor([0])
        
        # init losses D
        self._loss_df_real  = torch.cuda.FloatTensor([0])
        self._loss_df_fake = torch.cuda.FloatTensor([0])
        self._loss_df_gp = torch.cuda.FloatTensor([0])
        
        

    def set_train(self):
        self._Gf.train()
        self._Df.train()
       
        self._is_train = True

    def set_eval(self):
        self._Gf.eval()
        self._Df.eval()
       
        self._is_train = False


    def optimize_parameters(self, train_generator=True, save_imgs = False):
        if self._is_train:

            loss_D, real_samples_fg, fake_samples_fg, real_samples_mask, fake_samples_mask = self._forward_D()
            self._optimizer_Df.zero_grad()
            loss_D.backward()
            self._optimizer_Df.step()

            self._loss_df_gp = torch.cuda.FloatTensor([0])

            for t in range(self._T - 1):
                self._loss_df_gp = self._loss_df_gp + self._gradient_penalty_Df(real_samples_fg[t], fake_samples_fg[t], is_fg = True)* self._opt.lambda_Df_gp
                self._loss_df_gp = self._loss_df_gp + self._gradient_penalty_Df(real_samples_mask[t], fake_samples_mask[t], is_fg = False)* self._opt.lambda_Df_gp

            loss_D_gp = self._loss_df_gp
            loss_D_gp.backward()
            self._optimizer_Df.step()

            
            # train G
            self._Gf.reset_params()
            
            # if train_generator:
            if(train_generator):
                
                loss_G = self._forward_G()
                self._optimizer_Gf.zero_grad()
                loss_G.backward()
                self._optimizer_Gf.step()
                self._Gf.reset_params()
                self._print_losses()


    def _forward_G(self):

        self._loss_g_fg = torch.cuda.FloatTensor([0])
        self._loss_g_fb_rec = torch.cuda.FloatTensor([0])

        for t in range(self._T - 1):
            self._move_inputs_to_gpu(t)

            # generate fake samples
            Inext_fake_fg, mask_next_fg = self._generate_fake_samples(t)
            self._curr_f = Inext_fake_fg 
            
            # Fake fgs
            d_fake_fg = self._Df(Inext_fake_fg, is_fg=True)
            self._loss_g_fg = self._loss_g_fg + self._compute_loss_D(d_fake_fg, False) * self._opt.lambda_Gf_prob_fg
            
            # Fake masks
            d_fake_mask = self._Df(mask_next_fg, is_fg=False)
            self._loss_g_fg = self._loss_g_fg + self._compute_loss_D(d_fake_fg, False) * self._opt.lambda_Gf_prob_mask
            
            # Fake images
            self._loss_g_fb_rec = self._loss_g_fb_rec + self._criterion_Gs_rec(self._next_frame_imgs_ori*mask_next_fg + (1-mask_next_fg)*-1.0, Inext_fake_fg) * self._opt.lambda_rec

        


        return self._loss_g_fb_rec + self._loss_g_fg
            

    def _generate_fake_samples(self, t):
        Inext_fake_fg, mask_next_fg = self._Gf(self._curr_f, self._curr_OFs)
        self._visual_masks.append(mask_next_fg)
        self._visual_fgs.append(Inext_fake_fg)
        return Inext_fake_fg, mask_next_fg

    
    def _generate_fake_samples_test(self, t):
        Inext_fake_fg, mask_next_fg = self._Gf(self._curr_f, self._curr_OFs)
        return Inext_fake_fg, Inext_fake_bg, mask_next_fg


    def forward(self, T):
        fgs = []
        bgs = []
        fakes = []
        masks = []
        with torch.no_grad():
            for t in range(T-1):
                # self._curr_OFs = self._OFs[t].cuda()
                self._move_inputs_to_gpu(t)
                Inext_fake, Inext_fake_fg, Inext_fake_bg, mask_next_fg = self._generate_fake_samples_test(t)
                self._curr_f = Inext_fake_fg 
                self._curr_b = Inext_fake_bg
                fgs.append(Inext_fake_fg)
                bgs.append(Inext_fake_bg)
                fakes.append(Inext_fake)
                masks.append(mask_next_fg)
        return fgs, bgs, fakes, masks


    def _forward_D(self):

        real_samples_fg = []
        fake_samples_fg = []
     
        fake_samples_mask = []
        real_samples_mask = []

        self._loss_df_real = torch.cuda.FloatTensor([0])
        self._loss_df_fake = torch.cuda.FloatTensor([0])
        
       

        for t in range(self._T-1):    # 0, 1, 2, 3,..., T-2,
          
            self._move_inputs_to_gpu(t)

            # generate fake samples
            Inext_fake_fg, mask_next_fg = self._generate_fake_samples(t)

            Inext_fake_fg = Inext_fake_fg.detach()
            self._curr_f = Inext_fake_fg
            real_samples_fg.append(self._first_fg)
            fake_samples_fg.append(Inext_fake_fg)

            mask_next_fg = mask_next_fg.detach()
            real_samples_mask.append(self._transformed_mask)
            fake_samples_mask.append(mask_next_fg)

            # Df(real_fg) & Df(fake_fg)
            d_real_fg = self._Df(self._first_fg, is_fg=True)
            self._loss_df_real = self._loss_df_real + self._compute_loss_D(d_real_fg, True) * self._opt.lambda_Df_prob_fg
            d_fake_fg = self._Df(Inext_fake_fg, is_fg=True)
            self._loss_df_fake = self._loss_df_fake + self._compute_loss_D(d_fake_fg, False) * self._opt.lambda_Df_prob_fg

            # Df(real_mask) & Df(fake_mask)
            d_real_mask = self._Df(self._transformed_mask, is_fg=False)
            self._loss_df_real = self._loss_df_real + self._compute_loss_D(d_real_mask, True) * self._opt.lambda_Df_prob_mask
            d_fake_mask = self._Df(mask_next_fg, is_fg=False)
            self._loss_df_fake = self._loss_df_fake + self._compute_loss_D(d_fake_mask, False) * self._opt.lambda_Df_prob_mask

        return self._loss_df_fake + self._loss_df_real, real_samples_fg, fake_samples_fg, real_samples_mask, fake_samples_mask


    def _gradient_penalty_Df(self, real_samples, fake_samples, is_fg=True):
        # interpolate sample
        # real_samples are always the first foregrounds (B, 3, H, W)
        # Fake samples are the foregrounds generated in each timestep
        alpha = torch.rand(self._opt.batch_size, 1, 1, 1).cuda().expand_as(real_samples)
        interpolated = Variable(alpha * real_samples.data + (1 - alpha) * fake_samples.data, requires_grad=True)
        interpolated_prob = self._Df(interpolated, is_fg)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        loss_d_gp = torch.mean((grad_l2norm - 1) ** 2)

        return loss_d_gp


    

    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)


    def get_losses(self):
        losses = {}
        
        losses['loss_df_real'] = self._loss_df_real.item()
        losses['loss_df_fake'] = self._loss_df_fake.item()
        losses['loss_df_gp'] =  self._loss_df_gp.item()
       
        losses['loss_g_fg'] = self._loss_g_fg.item()
        losses['loss_g_fb_rec'] = self._loss_g_fb_rec.item()
        loss_df = self._loss_df_fake.item() + self._loss_df_real.item() + self._loss_df_gp.item()
        losses['loss_df'] = loss_df
     
        return losses

    def get_imgs(self):
        visuals = {}
        visuals['masks'] = []
        visuals['fgs'] = []
       
        r = np.random.randint(0,self._opt.batch_size, 1) # batch
        for i in range(self._T - 1):
            lamascara = self._visual_masks[i][r,:,:,:].expand_as(self._curr_f[r,:,:,:].cpu().detach()).cpu().detach()
            visuals['masks'].append(util.tensor2im(lamascara))
            visuals['fgs'].append(tensor2im(self._visual_fgs[i][r,:,:,:].cpu().detach()))
          
        return visuals
    
    def _print_losses(self):
        print("MSE =" + "{:.2f}".format(self._loss_g_fb_rec.item()))
        discr_f = self._loss_df_fake.item() + self._loss_df_real.item()
        print("Df(fake) - Df(real) = ", "{:.2f}".format(discr_f) )
       
    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_Gf = self._opt.lr_Gf / self._opt.nepochs_decay
        self._current_lr_Gf -= lr_decay_G
        for param_group in self._optimizer_Gf.param_groups:
            param_group['lr'] = self._current_lr_Gf
        print('update G learning rate: %f -> %f' %  (self._current_lr_Gf + lr_decay_Gf, self._current_lr_Gf))

        lr_decay_Gf = self._opt.lr_Gf / self._opt.nepochs_decay
        self._current_lr_Gf -= lr_decay_G
        for param_group in self._optimizer_Gf.param_groups:
            param_group['lr'] = self._current_lr_Gf
        print('update G learning rate: %f -> %f' %  (self._current_lr_Gf + lr_decay_Gf, self._current_lr_Gf))

        # update learning rate D
        lr_decay_Df = self._opt.lr_Df / self._opt.nepochs_decay
        self._current_lr_Df -= lr_decay_Df
        for param_group in self._optimizer_Df.param_groups:
            param_group['lr'] = self._current_lr_Df
        print('update D learning rate: %f -> %f' %  (self._current_lr_Df + lr_decay_Df, self._current_lr_Df))

        lr_decay_Db = self._opt.lr_Db / self._opt.nepochs_decay
        self._current_lr_Db -= lr_decay_Db
        for param_group in self._optimizer_Db.param_groups:
            param_group['lr'] = self._current_lr_Db
        print('update D learning rate: %f -> %f' %  (self._current_lr_Db + lr_decay_Db, self._current_lr_Db))
    
    def save(self, label):
        # save networks
        self._save_network(self._Gf, 'Gf', label)
        
        self._save_network(self._Df, 'Df', label)
  

        # save optimizers
        self._save_optimizer(self._optimizer_Gf, 'Gf', label)
      
        self._save_optimizer(self._optimizer_Df, 'Df', label)
       

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._Gf, 'Gf', load_epoch)
        
        if self._is_train:
            # load D
            self._load_network(self._Df, 'Df', load_epoch)
            # load optimizers
            self._load_optimizer(self._optimizer_Gf, 'Gf', load_epoch)
            self._load_optimizer(self._optimizer_Df, 'Df', load_epoch)
