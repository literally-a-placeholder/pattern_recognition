# number of neurons in the hidden layer
hidden_dim = 100

# learning rate
learning_rate = 0.005

# number of training epochs
n_epochs = 10


import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

path = '../datasets/mnist-permutated-png-format/mnist'
train_paths = []
val_paths = []
test_paths = []
for i in range(10):
    trp = os.listdir(path+'/train/'+str(i))
    for trpath in trp:
        train_paths = train_paths + [path+'/train/'+str(i)+'/'+trpath]
    vap = os.listdir(path+'/val/'+str(i))
    for vapath in vap:
        val_paths = val_paths + [path+'/val/'+str(i)+'/'+vapath]
    tep = os.listdir(path+'/test/'+str(i))
    for tepath in tep:
        test_paths = test_paths + [path+'/test/'+str(i)+'/'+tepath]
print(train_paths[0:5])

n = len(train_paths)
train_images = np.zeros((n, 28*28), dtype=np.int)
train_labels = np.zeros(n, dtype=np.int)
for i in range(n):
    ip = train_paths[i]
    image = Image.open(ip).convert('L')
    npimage = np.array(image).reshape(28*28)
    train_images[i,:] = npimage
    train_labels[i] = ip[-5]
    
n = len(val_paths)
val_images = np.zeros((n, 28*28), dtype=np.int)
val_labels = np.zeros(n, dtype=np.int)
for i in range(n):
    ip = val_paths[i]
    image = Image.open(ip).convert('L')
    npimage = np.array(image).reshape(28*28)
    val_images[i,:] = npimage
    val_labels[i] = ip[-5]
    
n = len(test_paths)
test_images = np.zeros((n, 28*28), dtype=np.int)
test_labels = np.zeros(n, dtype=np.int)
for i in range(n):
    ip = test_paths[i]
    image = Image.open(ip).convert('L')
    npimage = np.array(image).reshape(28*28)
    test_images[i,:] = npimage
    test_labels[i] = ip[-5]
    
print(train_images.shape, train_labels.shape)
print(val_images.shape, val_labels.shape)
print(test_images.shape, test_labels.shape)

class DigitDataset(Dataset):
    def __init__(self, features, labels):
        super(DigitDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.transforms = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        image = self.features[index,:]
        image = image.reshape(28,28,1)
        image = self.transforms(image)
        image = image.float()
        label = self.labels[index]
        label = label.astype(np.int64)
        return (image, label)

train_dataset = DigitDataset(train_images, train_labels)
val_dataset = DigitDataset(val_images, val_labels)
test_dataset = DigitDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.main = nn.Sequential(nn.Linear(28*28, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 10))
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.main(out)
        return out

model = MLPModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train():
    model.train()
    train_loss = 0
    for iteration, (images, labels) in enumerate(train_loader):
        output = model(images)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = train_loss / len(train_loader)
    return average_loss

def validation():
    model.eval()
    val_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = loss_fn(output, labels)
            val_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()
            
    accuracy = 100.0 * n_correct / len(val_loader.dataset)
    average_loss = val_loss / len(val_loader)
    return val_loss, accuracy

def test():
    model.eval()
    test_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()
            
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader)
    return test_loss, accuracy

train_losses = []
val_losses = []
val_accuracy = []
for epoch in range(n_epochs):
    train_loss = train()
    train_losses.append(train_loss)
    val_loss, accuracy = validation()
    val_losses.append(val_loss)
    val_accuracy.append(accuracy)
    print('Epoch {}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.1f}%'.format(epoch+1, train_loss, val_loss, accuracy))

plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Losses')
plt.plot(np.arange(1, n_epochs+1), train_losses)
plt.plot(np.arange(1, n_epochs+1), val_losses)
plt.legend(['training', 'validation'])

test_loss, accuracy = test()
print('Test loss: {:.4f}, accuracy: {:.1f}%'.format(test_loss, accuracy))