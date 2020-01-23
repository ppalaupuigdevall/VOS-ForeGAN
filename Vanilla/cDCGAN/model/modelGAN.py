from torch import nn
from torch.autograd.variable import Variable
import sys
sys.path.insert(0, '/home/marina/GANs/')
from Vanilla.cDCGAN.networks.generator import Generator
from Vanilla.cDCGAN.networks.discriminator import Discriminator


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

    # Generate Fake Samples sampling from Random Noise
    def generate_samples(self, c, batch_size, z=None):
        if z is None:
            z = norm_noise(batch_size)
        return self._generator(z, c)

    # Optimize Model
    def step_optimization(self, real_samples, real_labels):
        # Generate Fake Labels and Samples
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, real_samples.size(0)))).cuda()  # size: (BS)
        fake_samples = self.generate_samples(fake_labels, real_samples.size(0))  # size: (BS, 1, 32, 32)

        # Optimize Generator
        # Optimize Discriminator
        loss_d = self._step_opt_d(real_samples, real_labels, fake_samples.detach(), fake_labels.detach())
        loss_g = self._step_opt_g(fake_samples, fake_labels)
        # Detach() detaches the output from the computational graph, no gradient will be backpropagated along this variable.

        return loss_g, loss_d

    def _step_opt_g(self, fake_samples, fake_labels):
        # 1. Reset gradients
        self._opt_g.zero_grad()

        # 2. Evaluate fake samples
        validity = self._discriminator(fake_samples, fake_labels)

        # 3. Calculate error and backpropagate
        loss = self._criterion(validity, self._label_real)
        loss.backward()

        # 4. Update Weights
        self._opt_g.step()

        return loss.item()

    def _step_opt_d(self, real_samples, real_labels, fake_samples, fake_labels):
        # 1. Reset gradients
        self._opt_d.zero_grad()

        # 2. Discriminate real samples
        validity_real = self._discriminator(real_samples, real_labels)
        loss_real = self._criterion(validity_real, self._label_real)

        # 3. Discriminate fake samples
        validity_fake = self._discriminator(fake_samples, fake_labels)
        loss_fake = self._criterion(validity_fake, self._label_fake)

        # 4. Total discriminator loss and backpropagate
        loss = (loss_real + loss_fake) / 2
        loss.backward()

        # 5. Update weights
        self._opt_d.step()
        return loss_real.item(), loss_fake.item()
