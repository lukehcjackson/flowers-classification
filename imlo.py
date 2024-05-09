#IMLO COURSEWORK

import torch
import torchvision

########  LOAD THE DATASET   ############
#train: 1030, validation: 1030, test: 6149, total:

batch_size = 4

train_set = torchvision.datasets.Flowers102(root='./data', split="train", download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

validation_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(len(train_set))
print(len(validation_set))
print(len(test_set))


