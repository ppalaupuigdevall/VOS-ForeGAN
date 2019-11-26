import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np
import math
import sys
sys.path.insert(0, '/Users/marinaalonsopoal/PycharmProjects/GANs/data')
from mnist import MNIST

# Create a dataloader for the MNIST dataset
batch_size = 100
data_loader = torch.utils.data.DataLoader(MNIST(), batch_size=batch_size, shuffle=True)

num_epochs = 5
num_val_samples = 25
z_val = norm_noise(num_val_samples)
model = Model(batch_size)

for epoch in range(num_epochs):
    # Train epoch
    for n_batch, (real_samples,_) in enumerate(data_loader):

        # Prepare batch data
        real_samples = Variable(real_samples).cuda()

        # Update model weights
        loss_g, loss_d = model.step_optimization(real_samples)

        # Show current loss
        if (n_batch) % 10 == 0:
            print(f"epoch: {epoch}/{num_epochs}, batch: {n_batch}/{len(data_loader)}, G_loss: {loss_g}, D_loss: {loss_d}")

        # Show fake samples
        if (n_batch) % 100 == 0:
            val_fake_samples = model.generate_samples(num_val_samples, z=z_val).data.cpu()
            display_batch_images(val_fake_samples)
