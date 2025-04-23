import torch
from matplotlib.pyplot import xlabel, legend
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    """生成人工合成的线性回归数据集
    
    使用给定的权重向量w和偏置b，生成带有随机噪声的线性回归数据。
    生成公式为: y = Xw + b + noise
    
    Args:
        w (torch.Tensor): 权重向量，决定了特征的数量和对应的真实权重
        b (float): 偏置值
        num_examples (int): 需要生成的样本数量
    
    Returns:
        tuple: 返回一个元组 (features, labels)
            - features (torch.Tensor): 形状为(num_examples, len(w))的特征矩阵
            - labels (torch.Tensor): 形状为(num_examples, 1)的标签向量
    
    Example:
        >>> # 生成具有2个特征的1000个样本
        >>> true_w = torch.tensor([2.0, -3.4])
        >>> true_b = 4.2
        >>> features, labels = synthetic_data(true_w, true_b, 1000)
        >>> print(features.shape)  # torch.Size([1000, 2])
        >>> print(labels.shape)    # torch.Size([1000, 1])
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """将数据转换为PyTorch数据迭代器
    
    Args:
        data_arrays (tuple of torch.Tensor): 输入数据数组，可以是(features, labels)这样的元组
        batch_size (int): 批量大小
        is_train (bool, optional): 是否为训练模式. 默认为True
            - 如果为True，数据会被随机打乱
            - 如果为False，数据会按顺序返回
    
    Returns:
        torch.utils.data.DataLoader: PyTorch数据加载器，可以按批次返回数据
    
    Example:
        >>> features = torch.randn(1000, 2)
        >>> labels = torch.randn(1000, 1)
        >>> train_iter = load_array((features, labels), batch_size=32)
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels=None):
    """获取Fashion-MNIST数据集的文本标签
    
    Args:
        labels: 标签索引列表或None。如果为None，返回所有标签列表
    
    Returns:
        list: 文本标签列表
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    if labels is None:
        return text_labels
    return [text_labels[int(i)] for i in labels]

def show_images(X, rows, cols, titles=None):
    """显示Fashion-MNIST图像
    
    Args:
        X: 要显示的图像数据，形状为(n, 28, 28)
        rows: 显示的行数
        cols: 显示的列数
        titles: 每个图像的标题，可选
    
    Example:
        >>> X = X.reshape(18, 28, 28)  # 调整数据形状
        >>> show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))
    """
    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[i].numpy() if torch.is_tensor(X[i]) else X[i], cmap='gray')
        plt.axis('off')
        if titles:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)



def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter, device=None):
    """计算在指定数据集上模型的精度
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 计算设备，如果为None则使用net的设备
    
    Returns:
        float: 模型在数据集上的精度
    """
    if device is None and isinstance(net, torch.nn.Module):
        device = next(net.parameters()).device
    
    net.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    
    with torch.no_grad():
        for X, y in data_iter:
            if device is not None:
                X = X.to(device)
                y = y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

def train_epoch_single(net, train_iter, loss, updater):
    """训练模型一个epoch"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # metric[0]:
    # 返回loss/样本总数， 分类正确/ 样本总数
    return metric[0] / metric[2], metric[1] / metric[2]

def train_epoch_visual(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并在同一窗口动态显示训练进度"""
    # 存储训练历史
    train_losses = []
    train_accs = []
    test_accs = []
    epochs = []
    
    # 开启交互模式，动态更新
    plt.ion()

    # 训练循环
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Training Progress')
    ax.set_xlim(0, num_epochs + 1)  # 在最大值上增加一点余量
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)

    for epoch in range(num_epochs):
        # 训练和评估
        train_metrics = train_epoch_single(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        
        # 收集数据
        epochs.append(epoch + 1)
        train_losses.append(float(train_metrics[0]))
        train_accs.append(float(train_metrics[1]))
        test_accs.append(float(test_acc))


        # 打印信息
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')


        if epoch % 1 != 0: continue
        # 更新线条数据

        ax.plot(epochs, train_losses, 'r-', linewidth=2, label='Train Loss')
        ax.plot(epochs, train_accs, 'g-', linewidth=2, label='Train Acc')
        ax.plot(epochs, test_accs, 'b--', linewidth=2, label='Test Acc')

        # 设置图形属性

        # 刷新
        ax.relim()  # 重新计算轴的界限
        ax.autoscale_view()  # 自动调整视图
        fig.canvas.draw()  # 重绘整个图形
        fig.canvas.flush_events()  #
        fig.show()
        plt.pause(0.01)  # 短暂暂停以确保图形更新



    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_acc': test_accs
    }