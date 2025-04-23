import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import utils.mlutils as mlutils
from torch import nn
import matplotlib.pyplot as plt

#
#多层感机
#


trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size=18, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=18, shuffle=False)

net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
net.apply(mlutils.init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

mlutils.train_epoch_visual(net, train_iter, test_iter, loss, num_epochs, trainer)

