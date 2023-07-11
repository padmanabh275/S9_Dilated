import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def model_summary(model, input_size):
    summary(model, input_size=input_size)

def get_cifar_transform():
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22),], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return train_transforms, test_transforms

def get_cifar_dataset(type, transform):

    if type == 'train':
        return datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    else:
        return datasets.CIFAR10('../data', train=False, download=True, transform=transform)

def get_data_loader(ds, kwargs):
    return torch.utils.data.DataLoader(ds, **kwargs)

class PPipeline:
    def __init__(self, model, device):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        
        # self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

        self.model = model
        self.device = device

    def GetCorrectPredCount(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self, train_loader, optimizer, criterion):
        self.model.train()
        pbar = tqdm(train_loader)
        
        train_loss = 0
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            # Predict
            pred = self.model(data)
            
            # Calculate loss
            loss = criterion(pred, target)
            train_loss+=loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)
            
            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))

    def test(self, test_loader, criterion):
        self.model.eval()
        
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
        
                output = self.model(data)
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
        
                correct += self.GetCorrectPredCount(output, target)
        
        
        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)
        
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def print_performance(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")