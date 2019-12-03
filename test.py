from utils import *
import matplotlib.pyplot as plt

batch_size = 1
# generator = torch.load('/home/marina/GANs/model/saved/generator.pth.tar')
model = torch.load('/home/marina/GANs/model/saved/model.pth.tar')
val_fake_samples = model.generate_samples(batch_size).data.cpu()
image_to_display = display_batch_images(val_fake_samples)

plt.imshow(image_to_display.permute(1, 2, 0))
plt.show()
