import torch
import sys
sys.path.insert(0, '/Users/marinaalonsopoal/PycharmProjects/GANs')
from data.mnist import mnist_dataset
from model.modelGAN import Model
from utils import *
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/cdcgan_train_ayuda')
import torchvision.utils as vutils


# Create a dataloader for the MNIST dataset
batch_size = 100
data_loader = torch.utils.data.DataLoader(mnist_dataset(), batch_size=batch_size, shuffle=True)

num_epochs = 10
num_val_samples = 100  # Number of images that will be displayed
z_val = norm_noise(num_val_samples)
model = Model(batch_size)
i = 0

for epoch in range(num_epochs):

    for n_batch, (real_samples, real_labels) in enumerate(data_loader):
        print(real_labels)
        # Prepare batch data
        real_samples = Variable(real_samples).cuda()  # size: (BS, 1, 32, 32)
        real_labels = Variable(real_labels).cuda()  # size: (BS)

        # Update model weights
        loss_g, loss_d = model.step_optimization(real_samples, real_labels)

        if n_batch % 100 == 0:
            i = i + 1
            print("Epoch %2d of %2d - Batch %2d of %2d - Gen.Loss:%.2f Disc.Loss:%.2f %.2f" % (epoch, num_epochs,
                  n_batch, len(data_loader), loss_g, loss_d[0], loss_d[1]))
            writer.add_scalars('data/losses', {'loss_gen': loss_g, 'loss_disc': (loss_d[0]+loss_d[1])/2}, i)

            # Display Fake Samples
            val_fake_labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()  # size: 100
            val_fake_samples = model.generate_samples(val_fake_labels, val_fake_labels.size(0),  z=z_val).data.cpu()

            f_samples = display_batch_images(val_fake_samples)
            writer.add_image('Generated Samples', f_samples, i)

            # Display Real Samples
            r_samples = display_batch_images(real_samples)
            writer.add_image('Real Samples', r_samples, i)

# Test
class_to_test = 0
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image = display_batch_images(test_sample)
writer.add_image('Test Sample', test_image, 1)

class_to_test = 1
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_1 = display_batch_images(test_sample)
writer.add_image('One', test_image_1, 1)

class_to_test = 2
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_2 = display_batch_images(test_sample)
writer.add_image('Two', test_image_2, 1)

class_to_test = 3
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_3 = display_batch_images(test_sample)
writer.add_image('Three', test_image_3, 1)

class_to_test = 4
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_4 = display_batch_images(test_sample)
writer.add_image('Four', test_image_4, 1)

class_to_test = 5
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_5 = display_batch_images(test_sample)
writer.add_image('Five', test_image_5, 1)

class_to_test = 6
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_6 = display_batch_images(test_sample)
writer.add_image('Six', test_image_6, 1)

class_to_test = 7
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_7 = display_batch_images(test_sample)
writer.add_image('Seven', test_image_7, 1)

class_to_test = 8
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_8 = display_batch_images(test_sample)
writer.add_image('Eight', test_image_8, 1)

class_to_test = 9
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_9 = display_batch_images(test_sample)
writer.add_image('Nine', test_image_9, 1)

writer.close()

# Save Model
# torch.save(model, '/home/marina/GANs/model/saved/model.pth.tar')
