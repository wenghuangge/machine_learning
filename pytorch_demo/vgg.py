import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 定义VGG16网络（为Fashion-MNIST调整）
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # 输入层调整：从1通道变为3通道
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        # 特征提取层 (与原始VGG16相同)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器部分 (调整为Fashion-MNIST的尺寸，输入应为512个特征)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 2. 数据准备
def load_data(batch_size=64):
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整大小以匹配VGG16
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,))  # Fashion-MNIST的统计值
    ])
    
    # 加载Fashion-MNIST数据集
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # 类别名称
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return train_loader, test_loader, classes

# 3. 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {i+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100.*correct/total

# 4. 评估函数
def evaluate(model, test_loader, device, classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 打印每个类别的准确率
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    accuracy = 100. * correct / total
    print(f'Overall Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 5. 可视化函数
def visualize_results(model, test_loader, device, classes, num_images=10):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size(0)):
                if images_so_far >= num_images:
                    return
                
                images_so_far += 1
                ax = plt.subplot(num_images//5 + 1, 5, images_so_far)
                ax.set_title(f'Pred: {classes[preds[j]]}\nTrue: {classes[labels[j]]}')
                ax.axis('off')
                
                # 从归一化恢复
                img = inputs[j].cpu().squeeze().numpy()
                img = (img * 0.3530) + 0.2861  # 使用Fashion-MNIST的均值和标准差
                plt.imshow(img, cmap='gray')
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_predictions.png')
    plt.show()

# 6. 主函数
def main():
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (torch.backends.mps.is_available()): device = "mps"

    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    
    # 加载数据
    train_loader, test_loader, classes = load_data(batch_size)
    
    # 创建模型
    model = VGG16(num_classes=10)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1)
    
    # 训练和评估
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        test_acc = evaluate(model, test_loader, device, classes)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # 学习率调度
        scheduler.step(test_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds")
        print("-" * 50)
    
    # 保存模型
    torch.save(model.state_dict(), 'vgg16_fashion_mnist.pth')
    
    # 绘制准确率变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fashion_mnist_accuracy_plot.png')
    plt.show()
    
    # 可视化一些预测结果
    visualize_results(model, test_loader, device, classes)

if __name__ == "__main__":
    main()
