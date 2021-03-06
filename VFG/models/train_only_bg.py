import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from models.models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
import torch.nn.functional as F
from data.dataset_davis import tensor2im
import cv2

class TrainGB(BaseModel):
    def __init__(self, opt):
        
        super(TrainGB, self).__init__(opt)
        self._name = 'train_gb'
        self._opt = opt
        self._extra_ch_Gb = 0

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
        self._warped_imgs = sample['warped_imgs'] # [w_0, w_1, ..., w_t-1]
       
        self._first_fg = sample['mask_f']
        # NOTE: If the generated samples look weird, comment the line below and replace it by self._first_b = sample['mask_b']
        #BUG The line below gnerates some shit that is not great
        self._first_bg = sample['mask_b']
        self._real_bg_patches = self._extract_real_patches(self._opt, self._first_fg, self._first_bg) # NOTE TODO This could be done for each t
        # NOTE: If the generated samples look weird, comment the line below
        # self._first_bg = sample['mask_b']*(1-sample['mask']) + sample['mask']*(torch.rand_like(sample['mask_b'])*2.0 - 1.0)
        self._mask = sample['mask']
        self._move_inputs_to_gpu()
        

    def _move_inputs_to_gpu(self):
        self._visual_masks = []
        self._visual_fgs = []
        self._visual_bgs = []
        self._visual_fakes = []
       
        self._first_bg = self._first_bg.cuda()
        self._real_bg_patches = self._real_bg_patches.cuda()
        self._mask = self._mask.cuda()
    

    def _extract_real_patches(self, opt, first_fg, first_bg):
        batch_size = opt.batch_size
        kh, kw, stride_h, stride_w = opt.kh, opt.kw, opt.stride_h, opt.stride_w
        kernel = torch.ones(1,3,kh,kw)
        output = F.conv2d(first_fg + 1.0, kernel, stride=(stride_h,stride_w))
        convsize = output.size()[-1]
        indexes = torch.le(output, 0.001)
        
        N = self._opt.num_patches
        nonzero_indexes = []
        nonzero_elements = [0] * batch_size
        for i in range(batch_size):
            cerveseta = indexes[i,0,:,:]
            nonz = torch.nonzero(cerveseta) # [nelem,2]
            nonzero_indexes.append(nonz)
            nelem = nonz.size()[0]
            nonzero_elements[i] = nelem
            N = min(nelem, N)
        

        image_patches = torch.zeros(batch_size,N,3,kh,kw)
        to_image_coords = torch.tensor([stride_h, stride_w]).expand((N,2))
        img_indexes = torch.zeros(batch_size, N, 2)

        for i in range(batch_size):
            random_integers = np.unique(np.random.randint(0,nonzero_elements[i],N))
            while(random_integers.shape[0]<N):
                random_integers = np.unique(np.random.randint(0,nonzero_elements[i],N))
            conv_indexes = nonzero_indexes[i][random_integers]
            img_indexes[i, :, :] = conv_indexes * to_image_coords

        for b in range(batch_size):
            for n in range(N):
                P1 = int(img_indexes[b,n,0])
                P2 = int(img_indexes[b,n,1])
                image_patches[b, n, :, :, :] = first_bg[b, :, P1 : P1 + kh, P2:P2+kw]
        print(image_patches.size())
        return image_patches

        
    def _extract_img_patches(self,x, ch=3, height=224, width=416, kh=60, kw=112,dh=3, dw=3):
        patches = x.unfold(2, kh, dh).unfold(3, kw, dw) # 2,3,9,9,60,112
        r = np.random.randint(0,patches.size()[2]*patches.size()[3], self._opt.num_patches)
        r = np.unravel_index(r,(patches.size()[2], patches.size()[3]))
        r = torch.from_numpy(np.asarray(r))
        patches = patches[:,:,r[0,:], r[1,:], :,:]
        patches = patches.view(-1, ch, kh, kw)
        return patches


    def _init_create_networks(self):

        self._Gb = self._create_generator_b()
        self._Gb.init_weights()
        if len(self._gpu_ids) > 1:
            self._Gb = torch.nn.DataParallel(self._Gb, device_ids=self._gpu_ids)
        self._Gb.cuda()

        self._Db = self._create_discriminator_b()
        self._Db.init_weights()
        if len(self._gpu_ids) > 1:
            self._Db = torch.nn.DataParallel(self._Db, device_ids=self._gpu_ids)
        self._Db.cuda()

    
    def _create_generator_b(self):
        return NetworksFactory.get_by_name('generator_wasserstein_gan_b_static_ACR_single_img', c_dim=self._extra_ch_Gb, T=self._opt.T)
    
    def _create_discriminator_b(self):
        return NetworksFactory.get_by_name('discriminator_wasserstein_gan')

    def _init_train_vars(self):
        self._current_lr_Gf = self._opt.lr_Gf
        self._current_lr_Gb = self._opt.lr_Gb
        self._current_lr_Df = self._opt.lr_Df
        self._current_lr_Db = self._opt.lr_Db

        # initialize optimizers
        self._optimizer_Gb = torch.optim.Adam(self._Gb.parameters(), lr=self._current_lr_Gb,
                                            betas=[self._opt.Gb_adam_b1, self._opt.Gb_adam_b2])
        self._optimizer_Db = torch.optim.Adam(self._Db.parameters(), lr=self._current_lr_Db,
                                             betas=[self._opt.Db_adam_b1, self._opt.Db_adam_b2])
        
    def _init_losses(self):
        # define loss functions
        self._rec_criterion = torch.nn.L1Loss().cuda()
        
        # init losses G
        
        self._loss_g_bg = torch.cuda.FloatTensor([0])
        self._loss_g_rec = torch.cuda.FloatTensor([0])
        # init losses D
        self._loss_db_real  = torch.cuda.FloatTensor([0])
        self._loss_db_fake  = torch.cuda.FloatTensor([0])
        self._loss_db_gp = torch.cuda.FloatTensor([0])

    def set_train(self):
        
        self._Gb.train()
        self._Db.train()
        self._is_train = True

    def set_eval(self):
        self._Gf.eval()
        self._Df.eval()
        self._Gb.eval()
        self._Db.eval()
        self._is_train = False


    def optimize_parameters(self, train_generator=True, save_imgs = False):
        if self._is_train:

            loss_D, real_samples_bg, fake_samples_bg = self._forward_D()
            self._optimizer_Db.zero_grad()
            loss_D.backward()
            self._optimizer_Db.step()

            self._loss_db_gp = self._gradient_penalty_Db(real_samples_bg[0], fake_samples_bg[0])               

            loss_D_gp = self._loss_db_gp
            loss_D_gp.backward()
            self._optimizer_Db.step()
                    
            # if train_generator:
            if(train_generator):
                
                loss_G = self._forward_G()
                self._optimizer_Gb.zero_grad()
                loss_G.backward()
                self._optimizer_Gb.step()
                self._print_losses()


    def _forward_G(self):

        self._move_inputs_to_gpu()
        # generate fake samples
        Inext_fake_bg = self._generate_fake_samples() 
        # Fake bgs
        patches_Inext_bg = self._extract_img_patches(Inext_fake_bg)
        d_fake_bg = self._Db(patches_Inext_bg)
        self._loss_g_bg = self._compute_loss_D(d_fake_bg, False)
        self._loss_g_rec = self._rec_criterion(Inext_fake_bg*(1-self._mask), self._first_bg*(1-self._mask)) * self._opt.lambda_rec
        return self._loss_g_bg + self._loss_g_rec
            

    def _generate_fake_samples(self):
        
        Inext_fake_bg = self._Gb(self._first_bg)
        self._visual_bgs.append(Inext_fake_bg)
        return Inext_fake_bg

    
    def _generate_fake_samples_test(self, t):
        Inext_fake_fg, mask_next_fg = self._Gf(self._curr_f, self._curr_OFs, self._curr_warped_imgs)
        Inext_fake_bg = self._Gb(self._curr_b)
        Inext_fake = (1 - mask_next_fg) * Inext_fake_bg + Inext_fake_fg
        return Inext_fake, Inext_fake_fg, Inext_fake_bg, mask_next_fg


    def forward(self, T):
        fgs = []
        bgs = []
        fakes = []
        masks = []
        with torch.no_grad():
            for t in range(T-1):
                self._curr_OFs = self._OFs[t].cuda()
                self._generate_fake_samples(t)
                Inext_fake, Inext_fake_fg, Inext_fake_bg, mask_next_fg = self._generate_fake_samples_test(t)
                self._curr_f = Inext_fake_fg 
                self._curr_b = Inext_fake_bg
                fgs.append(Inext_fake_fg)
                bgs.append(Inext_fake_bg)
                fakes.append(Inext_fake)
                masks.append(mask_next_fg)
        return fgs, bgs, fakes, masks


    def _forward_D(self):
        
        real_samples_bg = []
        fake_samples_bg = []

        self._move_inputs_to_gpu()
        # generate fake samples
        Inext_fake_bg = self._generate_fake_samples()
        Inext_fake_bg = Inext_fake_bg.detach()

        # Db(real_bg_patches) & Db(fake_bg_patches)
        paches_bg_real = self._real_bg_patches.view(-1,3,self._opt.kh, self._opt.kw) 
        real_samples_bg.append(paches_bg_real)
        d_real_bg = self._Db(paches_bg_real)
        self._loss_db_real = self._compute_loss_D(d_real_bg, True) * self._opt.lambda_Db_prob
        
        patches_bg_fake = self._extract_img_patches(Inext_fake_bg)
        fake_samples_bg.append(patches_bg_fake)
        d_fake_bg = self._Db(patches_bg_fake)
        self._loss_db_fake = self._compute_loss_D(d_fake_bg, False) * self._opt.lambda_Db_prob

        return self._loss_db_fake + self._loss_db_real, real_samples_bg, fake_samples_bg


    def _gradient_penalty_Db(self, real_samples, fake_samples):
        # interpolate sample
        alpha = torch.rand(self._opt.num_patches * self._opt.batch_size, 1, 1, 1).cuda().expand_as(real_samples)
        interpolated = Variable(alpha * real_samples.data + (1 - alpha) * fake_samples.data, requires_grad=True)
        interpolated_prob = self._Db(interpolated)

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
        
       
        losses['loss_db_real'] = self._loss_db_real.item()
        losses['loss_db_fake'] = self._loss_db_fake.item()
        losses['loss_db_gp'] = self._loss_db_gp.item()
        
        losses['loss_g_bg'] = self._loss_g_bg.item()
        
        loss_db = self._loss_db_fake.item() + self._loss_db_real.item() + self._loss_db_gp.item()
        losses['loss_db'] = loss_db
        return losses


    def get_imgs(self):
        visuals = {}
        visuals['bgs'] = []
        r = np.random.randint(0,self._opt.batch_size, 1) # batch
        
        visuals['bgs'].append(tensor2im(self._visual_bgs[0][r,:,:,:].cpu().detach()))
        return visuals
    
    def _print_losses(self):
        discr_b = self._loss_db_fake.item() + self._loss_db_real.item()
        print("Db(fake) - Db(real) = ", "{:.2f}".format(discr_b) )
        print("L1 = ", "{:.2f}".format(self._loss_g_rec.item()) )


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
        self._save_network(self._Gb, 'Gb', label)
        self._save_network(self._Db, 'Db', label)

        # save optimizers
        self._save_optimizer(self._optimizer_Gb, 'Gb', label)
        self._save_optimizer(self._optimizer_Db, 'Db', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        # self._load_network(self._Gf, 'Gf', load_epoch)
        self._load_network(self._Gb, 'Gb', load_epoch)
        
        if self._is_train:
            # load D
            # self._load_network(self._Df, 'Df', load_epoch)
            self._load_network(self._Db, 'Db', load_epoch)
            # load optimizers
            # self._load_optimizer(self._optimizer_Gf, 'Gf', load_epoch)
            self._load_optimizer(self._optimizer_Gb, 'Gb', load_epoch)
            # self._load_optimizer(self._optimizer_Df, 'Df', load_epoch)
            self._load_optimizer(self._optimizer_Db, 'Db', load_epoch)