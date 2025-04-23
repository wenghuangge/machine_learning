import torch
import torch.nn as nn
import torch.optim as optiom
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. 设置参数
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 数据预处理加载
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

print(train_dataset[0][0].shape)
train_loader = DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #第二个卷积层
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层
        self.fc = nn.Linear(in_features=32*7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.con2(x)
        x = x.view(x.size(0), -1)
        ouptput = self.fc(x)
        return ouptput

#训练和评估函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        #前向传播
        output = model(data)
        loss = criterion(output, target)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 打印训练进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch: {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')

    return train_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader), 100. * correct / total


# 5. 可视化函数
def plot_results(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 6. 主程序
def main():
    #创建模型
    model = CNN().to(DEVICE)

    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 训练循环
    print(f"Training on {DEVICE}")
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch + 1}/{EPOCHS}")

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, DEVICE)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # 绘制结果
    plot_results(train_losses, train_accs, test_losses, test_accs)

    # 保存模型
    #torch.save(model.state_dict(), 'mnist_cnn.pth')n_loss, train_acc = train(model, trainer_loader, criterion, optimizer, DEVICE)

if __name__ == '__main__':
    main()