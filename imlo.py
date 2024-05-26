#IMLO COURSEWORK

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    
    ########  LOAD THE DATASET   ############
    #train: 1020, validation: 1020, test: 6149, total: 8189

    #transform to convert from PIL image (0 - 1) to tensors with a range of (-1 - 1)
    transform = transforms.Compose(
        [transforms.Resize(size = img_size), #resizes the images to be all the same size -> determines resolution
        transforms.CenterCrop(img_crop), #crops the image from the center -> determines scale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #using known mean and standard deviation

    #DEFINE TRAIN / VALIDATION / TEST SETS (splits come from the dataset itself)
    train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
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

def train_network(net, train_loader, validation_loader, optimizer, criterion, learning_rate, decay, batch_size, mini_batch_size):

    for epoch in range(num_epochs):

        running_loss = 0.0

        #this loop will run 1020 / batch_size number of times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
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
                

        #validation - once per epoch
        validation_loss, validation_accuracy = validate_network(net, validation_loader, criterion)

        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) +
                #" Training Loss = " + str(running_loss/mini_batch_size) +
                " Validation Loss = " + str(round(validation_loss ,5)) +
                " Validation Accuracy = " + str((round(validation_accuracy, 5))*100) + "%" )

        learning_rate = learning_rate * (1 / (1 + decay * epoch))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        print("Learning rate:", learning_rate)

    print('Finished Training')

def validate_network(net, validation_loader, criterion):
    validation_loss = 0
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for i,data in enumerate(validation_loader):
            vimages, vlabels = data[0].to(device), data[1].to(device)
            voutputs = net(vimages)

            validation_loss += criterion(voutputs, vlabels).item()

            _, predicted = torch.max(voutputs.data, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    net.train(True)

    validation_loss = validation_loss / total
    validation_accuracy = correct / total
    return validation_loss, validation_accuracy

"""
    for i, data in enumerate(validation_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        validation_loss += criterion(outputs, labels).item()
        # Calculate probability
        ps = torch.exp(outputs)
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        validation_accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss, validation_accuracy.item()
"""
def test_network(net, test_loader):
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


#-------------MAIN------------------

#Load the dataset into dataloaders using the official splits
batch_size = 64 #the larger this is, the more epochs it takes for the loss to start decreasing (???)
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

learning_rate = 0.02 #0.02
#decay = learning_rate / num_epochs
decay = 0.001 #increase => faster lr decreases

#define a loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

#Define hyperparameters
mini_batch_size = 1 #each mini-batch is batch_size (64) x mini_batch_size images
num_epochs = 5

#Train the network
train_network(net, train_loader, validation_loader, optimizer, criterion, learning_rate, decay, batch_size, mini_batch_size)

#Test the network
test_network(net, test_loader)