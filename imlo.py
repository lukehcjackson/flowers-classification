#IMLO COURSEWORK

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

########  LOAD THE DATASET   ############
#train: 1020, validation: 1020, test: 6149, total: 8189

batch_size = 32
img_size = 32
img_crop = 32

#transform to convert from PIL image (0 - 1) to tensors with a range of (-1 - 1)
transform = transforms.Compose(
    [transforms.Resize(size = img_size), #resizes the images to be all the same size -> determines resolution
     transforms.CenterCrop(img_crop), #crops the image from the center -> determines scale
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #use different mean / std ?
#should this transformation only be applied to the training data? or should there be another transform including random rotation / cropping / scaling for training data?

#DEFINE TRAIN / VALIDATION / TEST SETS (splits come from the dataset itself)
train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

validation_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#print(len(train_set))
#print(len(validation_set))
#print(len(test_set))

# functions to show an image
#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
#
#dataiter = iter(train_loader)
#images, labels = next(dataiter)
##show images
#imshow(torchvision.utils.make_grid(images))
#print(labels)

#DEFINING AND TRAINING CONVOLUTIONAL NEURAL NETWORK

#training on the GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#Define CNN
class Net(nn.Module):
    #neural network as before, but modified to take 3-channel images
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 103) #second argument is number of classes in dataset

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#instantiate network
net = Net()
#go over all modules and set parameters and buffers to CUDA tensors
net.to(device)

#define a loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

mini_batch_size = (1020 / batch_size) // 5

#train the network
#loop over our data iterator, and feed the inputs to the network and optimise
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0

    #this loop will run 1020 / batch_size number of times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #FOR CPU: 
        #inputs, labels = data
        #FOR GPU:
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % mini_batch_size == 0:    #WHAT IS THIS DOING? CHANGING THIS DRAMATICALLY EFFECTS THE LOSS!!
            print("Epoch " + str(epoch+1) + " Batch " + str(i) + " : Loss = " + str(running_loss))
            running_loss = 0.0

print('Finished Training')

#make predictions for the whole dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        #FOR CPU: 
        #images, labels = data
        #FOR GPU:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 6149 test images: {100 * correct // total} %')

#TO DO:
"""
Use scipy to read included .mat files
Use the validation set to calculate accuracy as the network is training
Change the network architecture based on research
Tweak hyperparameters endlessly
"""