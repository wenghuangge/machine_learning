import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import utils.mlutils as mlutils
from torch import nn
import matplotlib.pyplot as plt

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
#
#打印数据 - 返回的是元组，[0]是数据， [1]是标签
print(mnist_train[0][0].shape)

#显示数据集
#mlutils.show_images(X.reshape(18, 28, 28), 2, 9, titles=mlutils.get_fashion_mnist_labels(y))

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(mlutils.init_weights)
#多分类使用交叉墒作为损失函数
loss = nn.CrossEntropyLoss()
#梯度下降方法
trainer = torch.optim.SGD(net.parameters(), lr=0.3)

X = mnist_test.data.float()
y = mnist_test.targets
print(X.shape, y.shape)

train_iter = data.DataLoader(mnist_train, batch_size=18, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=18, shuffle=False)

mlutils.train_epoch_visual(net, train_iter, test_iter, loss, 10, trainer)