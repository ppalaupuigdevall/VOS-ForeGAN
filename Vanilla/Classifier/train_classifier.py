import sys
import torch
import torch.optim as optim
sys.path.insert(0, '/home/marina/GANs/')
from Vanilla.mnist import mnist_dataset_train, mnist_dataset_test
from Vanilla.Classifier.classifier import Classifier
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/Classifier_train')

device = torch.device("cuda")

# Create a dataloader for the MNIST dataset
batch_size = 64
train_loader = torch.utils.data.DataLoader(mnist_dataset_train(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_dataset_test(), batch_size=batch_size, shuffle=True)

num_epochs = 10
model = Classifier().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
i = 0

log_interval = 100

for epoch in range(num_epochs):

    model.train()
    for n_batch, (samples, labels) in enumerate(train_loader):
        samples = samples.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicted = model(samples)
        loss = F.nll_loss(predicted, labels)
        loss.backward()
        optimizer.step()
        if n_batch % log_interval == 0:
            print("TRAIN Epoch %2d of %2d - Batch %2d of %2d - Loss: %.2f " % (epoch + 1, num_epochs,
                  n_batch + 100, len(train_loader), loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for n_batch, (samples, labels) in enumerate(test_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            predicted = model(samples)
            test_loss += F.nll_loss(predicted, labels, reduction='sum').item()  # sum up batch loss
            pred = predicted.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print("TEST Epoch %2d of %2d - Loss: %.2f  Accuracy: %.2f" % (epoch + 1, num_epochs, test_loss, accuracy))

    scheduler.step()


# Save Model
# torch.save(model, '/home/marina/GANs/Classifier/cDCGAN_model.pth.tar')
