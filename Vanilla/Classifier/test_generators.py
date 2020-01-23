import torch
import sys
sys.path.insert(0, '/home/marina/GANs/')
import torch.nn.functional as F

device = torch.device('cuda')

model_generator = torch.load('/home/marina/GANs/cDCGAN/cDCGAN_model_10epoch.pth.tar')
model_classifier = torch.load('/home/marina/GANs/Classifier/cDCGAN_model.pth.tar')

trials = int(1e3)
total_samples = trials * 100
loss = 0
correct = 0

with torch.no_grad():

    for t in range(trials):

        # Generate Fake Samples
        z_val = norm_noise(100)
        gen_labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()  # size: 100
        gen_samples = model_generator.generate_samples(gen_labels, gen_labels.size(0), z=norm_noise(gen_labels.size(0))).data.cpu()

        # Test Generated Samples on 99% Accuracy Classifier
        model_classifier.eval()
        gen_samples = gen_samples.to(device)
        gen_labels = gen_labels.to(device)
        predicted = model_classifier(gen_samples)
        loss += F.nll_loss(predicted, gen_labels, reduction='sum').item()  # sum up batch loss
        pred = predicted.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(gen_labels.view_as(pred)).sum().item()

        if t % 100 == 0:
            print('Trial %2d of %2d done!' % (t, trials))


loss /= total_samples
accuracy = 100. * correct / total_samples
print('Loss: %.2f' % loss)
print('Accuracy:', accuracy, '%')
