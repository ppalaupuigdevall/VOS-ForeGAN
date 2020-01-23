from torchvision import transforms, datasets


# Download and transform Dataset
def mnist_dataset():
    tf = transforms.Compose(
        [
            transforms.Resize(32, interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
    ])
    return datasets.MNIST(root='./data', train=True, transform=tf, download=True)

# Download and transform Dataset
def mnist_dataset_train():
    tf = transforms.Compose(
        [
            transforms.Resize(32, interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(root='./data', train=True, transform=tf, download=True)

# Download and transform Dataset
def mnist_dataset_test():
    tf = transforms.Compose(
        [
            transforms.Resize(32, interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(root='./data', train=False, transform=tf, download=True)
