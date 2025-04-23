# 线性模型DEMO

import torch
import matplotlib.pyplot as plt
from torch import nn
from utils.mlutils import synthetic_data, load_array, train_epoch_visual

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# plt.figure(figsize=(10, 6))
# plt.scatter(features[:, 1].numpy(), labels.numpy(), alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatter plot of features vs labels')
#plt.show()


batch_size = 10
train_iter = load_array((features, labels), batch_size)

#初始化神经网络
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0.0, 0.01)
net[0].bias.data.fill_(0)
#初始化损失函数
loss = nn.MSELoss()
#初始化梯SGD
trainer =  torch.optim.SGD(net.parameters(), lr=0.03)


test_iter = train_iter
train_epoch_visual(net, train_iter, test_iter, loss, 1000, trainer)


# 打印模型参数
print('\n模型参数对比：')
print(f'真实权重: {true_w}')
print(f'真实偏置: {true_b}')
print(f'估计权重: {net[0].weight.data.flatten()}')
print(f'估计偏置: {net[0].bias.data}')


