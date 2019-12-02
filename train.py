import torch
import sys
sys.path.insert(0, '/Users/marinaalonsopoal/PycharmProjects/GANs')
from data.mnist import mnist_dataset
from model.modelGAN import Model
from utils import *
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/gan_losses')
import torchvision.utils as vutils


# Create a dataloader for the MNIST dataset
batch_size = 100
data_loader = torch.utils.data.DataLoader(mnist_dataset(), batch_size=batch_size, shuffle=True)

num_epochs = 20
num_val_samples = 25  # Number of images that will be displayed
z_val = norm_noise(num_val_samples)
model = Model(batch_size)

i = 0
j = 0

for epoch in range(num_epochs):

    for n_batch, (real_samples, _) in enumerate(data_loader):

        # Prepare batch data
        real_samples = Variable(real_samples).cuda()

        # Update model weights
        loss_g, loss_d = model.step_optimization(real_samples)

        # Show current loss
        if n_batch % 10 == 0:
            i = i + 1
            print("Epoch %2d of%2d - Batch %2d of %2d - Gen.Loss:%.2f Disc.Loss:%.2f %.2f" % (epoch, num_epochs,
                  n_batch, len(data_loader), loss_g, loss_d[0], loss_d[1]))
            writer.add_scalar('data/loss_g', loss_g, i)
            writer.add_scalars('data/loss_d', {'loss_real': loss_d[0], 'loss_fake': loss_d[1]}, i)

        # Show fake samples
        if n_batch % 100 == 0:
            j = j + 1

            val_fake_samples = model.generate_samples(num_val_samples, z=z_val).data.cpu()
            image_to_display = display_batch_images(val_fake_samples)
            writer.add_image('Generated Samples', image_to_display, j)
            
            r_samples = display_batch_images(real_samples)
            writer.add_image('Real Samples', r_samples, j)


