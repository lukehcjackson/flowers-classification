#IMLO COURSEWORK

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import datetime

def load_dataset():
    
    ########  LOAD THE DATASET   ############
    #train: 1020, validation: 1020, test: 6149, total: 8189

    #transform to convert from PIL image (0 - 1) to tensors with a range of (-1 - 1)
    standardTransform = transforms.Compose(
        [transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize(size = img_size), #resizes the images to be all the same size -> determines resolution
        transforms.CenterCrop(img_crop), #crops the image from the center -> determines scale
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #using known mean and standard deviation
    
    trainingTransform = transforms.Compose(
        [transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.75, 1.25)),
        #transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=img_crop, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #DEFINE TRAIN / VALIDATION / TEST SETS (splits come from the dataset itself)
    unmodified_train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=standardTransform)
    transformed_train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=trainingTransform)
    #training data set is really small so we want to increase the size of the training set => concatenate untransformed and transformed data
    #this doubles the size of our training data
    train_set = torch.utils.data.ConcatDataset([unmodified_train_set, transformed_train_set])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True, transform=standardTransform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=standardTransform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader

def define_network():
    #Define CNN
    class Net(nn.Module):
        #neural network as before, but modified to take 3-channel images
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
            self.fc = nn.Linear(in_features = (int(img_crop / 4) * int(img_crop / 4) * 16), out_features = 102)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.fc(x.reshape(x.shape[0], -1))
            return x
        
    return Net()

def train_network(net, train_loader, validation_loader, optimizer, criterion, learning_rate, decay, mini_batch_size):

    for epoch in range(num_epochs):

        running_loss = 0.0

        #each iteration is a mini-batch of 'batch_size' (64) images
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass - put the inputs through the network
            outputs = net(inputs)
            #calculate loss
            loss = criterion(outputs, labels)
            #perform a backwards pass using that loss - calculate gradients
            loss.backward()
            #optimiser makes a 'step' - update parameters based on gradients
            optimizer.step()

            #add loss to running loss (reset every (batch_size x mini_batch_size) images)
            running_loss += loss.item()

            #reset the running loss every so often
            #if i % mini_batch_size == 0:
                #print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + " [" + str(i * batch_size) + "/2040]" + " : Loss = " + str(running_loss))
                #running_loss = 0.0
                

        #validation - once per epoch
        validation_loss, validation_accuracy = validate_network(net, validation_loader, criterion)

        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) +
                " Validation Loss = " + str(round(validation_loss ,5)) +
                " Validation Accuracy = " + str((round(validation_accuracy, 5))*100) + "%" )

        #apply a time-based learning rate schedule
        learning_rate = learning_rate * (1 / (1 + decay * epoch))
        #update the actual learning rate in the model
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        print("Learning rate:", learning_rate)

    print('Finished Training')

def validate_network(net, validation_loader, criterion):
    validation_loss = 0
    correct = 0
    total = 0

    net.eval()

    #very similar to final testing loop, but on the validation set
    with torch.no_grad():
        for i,data in enumerate(validation_loader):
            vimages, vlabels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            voutputs = net(vimages)

            #add to loss
            validation_loss += criterion(voutputs, vlabels).item()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(voutputs.data, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    net.train(True)

    validation_loss = validation_loss / total
    validation_accuracy = correct / total
    return validation_loss, validation_accuracy

def test_network(net, test_loader):
    #make predictions for the whole dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 6149 test images: {100 * round(correct / total,1)} %')


#-------------MAIN------------------

#Time the model
startTime = datetime.datetime.now()

#Load the dataset into dataloaders using the official splits
batch_size = 64
img_size = 256
img_crop = 224 
train_loader, validation_loader, test_loader = load_dataset()

#Define the device we are using
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#Define and instantiate network
net = define_network()
net.to(device)

learning_rate = 0.01 #0.02
#decay = learning_rate / num_epochs
decay = 0.0001 #increase => faster lr decreases

#define a loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

#Define hyperparameters
mini_batch_size = 1 #each mini-batch is batch_size (64) x mini_batch_size images
num_epochs = 100

#Train the network
train_network(net, train_loader, validation_loader, optimizer, criterion, learning_rate, decay, mini_batch_size)

#Test the network
test_network(net, test_loader)

#Display total time to run
endTime = datetime.datetime.now()
timeDiff = startTime - endTime
print(timeDiff)