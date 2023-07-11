from utils import get_device
device = get_device()
print("Device:", device)
#Prepare Cifar dataset
from utils import get_cifar_transform, get_cifar_dataset

train_transform, test_transform = get_cifar_transform()

train_data = get_cifar_dataset('train', train_transform)
test_data = get_cifar_dataset('test', test_transform)

from utils import get_data_loader

batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = get_data_loader(test_data, kwargs)
train_loader = get_data_loader(train_data, kwargs)

import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torchvision
# show images
imshow(torchvision.utils.make_grid(images[:4]))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

from model import Net
from utils import model_summary

model = Net().to(device)
model_summary(model, input_size=(1, 28, 28))


from utils import PPipeline
import torch.nn.functional as F
import torch.optim as optim

pp = PPipeline(model, device)

from torch.optim.lr_scheduler import StepLR

model =  Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
EPOCHS = 50
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    # scheduler.step()
    test(model, device, test_loader)
pp.print_performance()

from torchviz import make_dot

batch = next(iter(train_loader))
model.eval()
yhat = model(batch[0].to(device)) # Give dummy batch to forward().

make_dot(yhat, params=dict(list(model.named_parameters()))).render("CIFAR10", format="png")