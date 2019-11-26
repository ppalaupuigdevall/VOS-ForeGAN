import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np
import math

class Model:

    # Prepare Model ---------------------------------------
    def __init__(self, batch_size):
        self._create_networks()
        self._create_optimizer()
        self._init_criterion(batch_size)

    def _create_networks(self):
        # Create Network
        self._generator = Generator()
        self._discriminator = Discriminator()

        # Define Weights
        self._generator.apply(init_weights)
        self._discriminator.apply(init_weights)

        # Pass to CUDA
        self._generator.cuda()
        self._discriminator.cuda()


    def _create_optimizer(self):
        # Generator Optimizer
        self._opt_g = torch.optim.Adam(self._generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Discriminator Optimizer
        self._opt_d = torch.optim.Adam(self._discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


    def _init_criterion(self, batch_size):
        # Binary Cross-Entropy (BCE) Loss
        self._criterion = nn.BCELoss()
        # Convention: Discriminator will output 1 if it thinks that the sample is real and 0 if fake
        self._label_real = Variable(torch.ones(batch_size, 1)).cuda()
        self._label_fake = Variable(torch.zeros(batch_size, 1)).cuda()

    # Generate Fake Samples ------------------------------------
    def generate_samples(self, batch_size, z=None):
        # Generate fake samples sampling from random noise
        if z is None:
            z = norm_noise(batch_size)
        return self._generator(z)

    # Optimize Model --------------------------------------------
    def step_optimization(self, real_samples):
        # Generate Fake Samples
        fake_samples = self.generate_samples(real_samples.size(0))

        # Optimize Generator and Discriminator
        loss_g = self._step_opt_g(fake_samples)
        loss_d = self._step_opt_d(real_samples, fake_samples.detach())
        # Detach() detaches the output from the computational graph, no gradient will be backproped along this variable.

        return loss_g, loss_d

    def _step_opt_g(self, fake_samples):
        # 1. Reset gradients
        self._opt_g.zero_grad()

        # 2. Evaluate fake samples
        estim_fake = self._discriminator(fake_samples)

        # 3. Calculate error and backpropagate
        loss = self._criterion(estim_fake, self._label_real)
        loss.backward()

        # 4. Update Weights
        self._opt_g.step()

        return loss.item()

    def _step_opt_d(self, real_samples, fake_samples):
        # 1. Reset gradients
        self._opt_d.zero_grad()

        # 2. Discriminate real samples
        estim_real = self._discriminator(real_samples)
        loss_real = self._criterion(estim_real, self._label_real)

        # 3. Discriminate fake samples
        estim_fake = self._discriminator(fake_samples)
        loss_fake = self._criterion(estim_fake, self._label_fake)

        # 4. Total discriminator loss and backpropagate
        loss = (loss_real + loss_fake) / 2
        loss.backward()

        # 5. Update weights
        self._opt_d.step()

        return loss_real.item(), loss_fake.item()