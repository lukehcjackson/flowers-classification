#IMLO COURSEWORK

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

########  LOAD THE DATASET   ############
#train: 1020, validation: 1020, test: 6149, total: 8189

batch_size = 32
img_size = 200
img_crop = 200

#transform to convert from PIL image (0 - 1) to tensors with a range of (-1 - 1)
transform = transforms.Compose(
    [transforms.Resize(size = img_size), #resizes the images to be all the same size -> determines resolution
     transforms.CenterCrop(img_crop), #crops the image from the center -> determines scale
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#should this transformation only be applied to the training data? or should there be another transform including random rotation / cropping / scaling for training data?

#DEFINE TRAIN / VALIDATION / TEST SETS (splits come from the dataset itself)
train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

validation_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(len(train_set))
print(len(validation_set))
print(len(test_set))

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
#show images
imshow(torchvision.utils.make_grid(images))