import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def extract_data(data, step):
    X = []
    Y = []
    for i in range(len(data) - step):
        X.append([a for a in data[i: i + step]])
        Y.append(data[i + step])
    X = np.array(X)
    # [样本数， 序列数量， 每个数据维度]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    print(X.shape, len(Y))
    return X, Y

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_file_path = os.path.join(current_dir, 'zgpa_train.csv')
data = pd.read_csv(data_file_path)
price = data.loc[:, 'close']

#归一化处理
price_norm = price / max(price)
print(price_norm.head())


fig1 = plt.figure(figsize=(8, 5))
plt.plot(price_norm)
plt.title('price')
plt.xlabel('Date')
plt.ylabel('price')
#plt.show()

time_step = 8
X, Y = extract_data(price_norm, 8)

# 定义RNN模型
class RNN(nn.Module):
    #input_size: 输入数据维度, 每个时间步只有一个特征（股票价格），所以设为1，如果有多个特征（如开盘价、收盘价、成交量等），这个值会相应增加
    #hidden_size: 隐藏层神经元数量，这决定了模型的容量和表达能力
    #num_layers: 隐藏层数量，RNN可以堆叠多个隐藏层，增加模型的深度
    #output_size: 输出数据维度， 我们只预测一个值（下一个时间点的股价），所以设为1
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层，输出预测值
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态 - 【层数， 批量大小， 隐藏数】
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        # 取RNN最后一个时间步的输出, rnn输出[batch_size, sequence_length, hidden_size]
        out = self.fc(out[:, -1, :])
        return out

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 计算训练集平均损失
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # 评估模型
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
    # 计算测试集平均损失
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.savefig(os.path.join(current_dir, 'loss_curve.png'))

# 预测
model.eval()
with torch.no_grad():
    # 用测试集的第一个样本进行预测
    test_seq = X_test[:50].to(device)
    preds = model(test_seq).cpu().numpy()
    actual = y_test[:50].numpy()

# 绘制预测结果
plt.figure(figsize=(10, 5))
plt.plot(actual, label='Actual Price')
plt.plot(preds, label='Predicted Price')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.savefig(os.path.join(current_dir, 'prediction_result.png'))

print("训练完成！结果已保存为图片。")



