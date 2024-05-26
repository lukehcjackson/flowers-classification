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

batch_size = 64 #the larger this is, the more epochs it takes for the loss to start decreasing (???)
img_size = 200 
img_crop = 200 
#https://stackoverflow.com/questions/57815801/what-defines-legitimate-input-dimensions-of-a-pytorch-nn

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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1)) #arg1 = layer1 arg2, arg2 = fc1 arg1 / kernelsize^2, arg3=kernelsize
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 103) #second argument is number of classes in dataset
        self.fc = nn.Linear(in_features = (int(200 / 4) * int(200 / 4) * 16), out_features = 102)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x

#instantiate network
net = Net()
#go over all modules and set parameters and buffers to CUDA tensors
net.to(device)

mini_batch_size = (1020 / batch_size) // 10 #num mini-batches = divisor + 1
#leaving this variable the same but REDUCING batch size => larger loss?
#leaving batch size the same but increasing the divisor => MAKES NO DIFFERENCE!!!!!!!!
num_epochs = 30

learning_rate = 0.02
#decay = learning_rate / num_epochs
decay = 0.001 #increase => faster lr decreases

#define a loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

#train the network
#loop over our data iterator, and feed the inputs to the network and optimise
for epoch in range(num_epochs):  # loop over the dataset multiple times

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
            print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + " [" + str(i * batch_size) + "/1020]" + " : Loss = " + str(running_loss))
            running_loss = 0.0

    learning_rate = learning_rate * (1 / (1 + decay * epoch))
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    print("Learning rate:", learning_rate)

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
Use scipy to read included .mat files????
Use the validation set to calculate accuracy as the network is training
Change the network architecture based on research
Tweak hyperparameters endlessly
"""