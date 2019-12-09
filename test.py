from utils import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/cdcgan_test')

batch_size = 25
z_val = norm_noise(batch_size)
model = torch.load('/home/marina/GANs/model/saved/model.pth.tar')

x = 45
writer.add_scalar('Bla', x, 1)

# Display zeros
class_to_test = 0
test_label = torch.LongTensor([class_to_test for _ in range(batch_size)]).cuda()
test_sample = model.generate_samples(test_label, 1, z=z_val).data.cpu()
test_image_0 = display_batch_images(test_sample)
writer.add_image('Zero', test_image_0, 1)

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
print('done')
