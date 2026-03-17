# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary
BATCH_SIZE = 8
# 1. 准备数据集.
def create_dataset():
    # 1. 获取训练集.
    # 参1: 数据集路径. 参2: 是否是训练集. 参3: 数据预处理 → 张量数据. 参4: 是否联网下载
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 2. 获取测试集.
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 3. 返回数据集.
    return train_dataset, test_dataset
# 2. 搭建(卷积)神经网络.
class ImageModel(nn.Module):
    #todo 1.初始化父类成员，搭建神经网络
    def __init__(self):
        # 1.1初始化父类成员
        super().__init__()
        # 1.2搭建神经网络
        # 第1个卷积层，输入3通道，输出6通道，卷积核大小3*3，步长1，填充0
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        # 第1个池化层，窗口大小2*2，步长2，填充0
        self.pool1 = nn.MaxPool2d(2,2,0)

        # 第2个卷积层，输入6通道，输出16通道，卷积核按大小3*3，步长1，填充0
        self.conv2 = nn.Conv2d(6,16,3,1,0)
        # 第2个池化层，窗口大小2*2，步长2，填充0
        self.pool2 = nn.MaxPool2d(2,2,0)

        # 第1个隐藏层（全连接层），输入：576，输出：120
        self.linear1 = nn.Linear(576,120)
        # 第2个隐藏层（全连接层），输入：120，输出：84
        self.linear2 = nn.Linear(120,84)
        # 第3个隐藏层（全连接层）→输出层，输入：84，输出：10
        self.output = nn.Linear(84,10)
    #todo 2.定义向前传播
    def forward(self,x):
        # 第1层：卷积层（加权求和）+激励层（激活函数）+池化层（降维）
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层：卷积层（加权求和）+激励层（激活函数）+池化层（降维）
        x = self.pool2(torch.relu(self.conv2(x)))

        # 3.1 展平数据，将多维数据变为一维数据(8,16,6,6)→(8,576)
        # 参1：样本数（行数），参2：特征数（列数），-1表示自动计算
        x = x.reshape(x.size(0),-1)   #8行576列
        # print(f'x.shape:{x.shape}')

        # 第3层：全连接层（加权求和）+激励层（激活函数）
        x = torch.relu(self.linear1(x))

        # 第4层：全连接层（加权求和）+激励层（激活函数）
        x = torch.relu(self.linear2(x))

        # 第5层：全连接层（加权求和）+输出层
        return self.output(x)      #后续用多分类交叉熵损失函数CrossEntropyLoss = Softmax()激活函数 + 损失计算
# 3. 模型训练.
def train(train_dataset):
    # 1.创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 2.创建模型对象
    model = ImageModel()
    # 3.创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 5.循环遍历epoch，开始每轮的训练动作
    # 5.1.定义变量，记录训练的总轮数
    epochs = 10
    # 5.2.遍历，完成每轮的所有批次的训练动作
    for epoch in range(epochs):
        # 5.2.1.定义变量，记录：总损失，总样本数据量，预测正确样本数，训练（开始）时间
        total_loss,total_samples,total_correct,start = 0.0,0,0,time.time()
        # 5.2.2.遍历数据加速器，获取到每批次的数据
        for x,y in dataloader:
            # 5.2.3.切换训练模式
            model.train()
            # 5.2.4.模型预测
            y_pred = model(x)
            # 5.2.5.计算损失
            loss = criterion(y_pred,y)
            # 5.2.6.梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.7.统计预测正确的样本个数
            total_correct += (torch.argmax(y_pred,dim=-1) == y).sum()
            # 5.2.8.统计当前批次的总损失  第1批平均损失 * 第1批样本个数
            total_loss += loss.item() * len(y) # 第1批总损失 + 第2批总损失 + ...
            # 5.2.9.统计当前批次的总样本个数
            total_samples += len(y)
        # 5.2.10.一轮训练结束，打印该轮训练信息
        print(f'轮次：{epoch+1}/{epochs}，总损失：{total_loss/total_samples:.4f}，准确率：{total_correct/total_samples:.4f}，训练时间：{time.time()-start:.4f}秒')
    # 6. 保存模型
    torch.save(model.state_dict(), 'model/image_model.pth')
# 4. 模型测试.
def evaluate(test_dataset):
    # 1.创建测试机数据加载器
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 2.创建模型对象
    model = ImageModel()
    # 3.加载模型参数
    model.load_state_dict(torch.load('model/image_model.pth'))  #pickle文件
    # 4.定义统计变量预测正确的样本个数，总样本个数
    total_correct,total_samples = 0,0
    # 5.循环遍历数据加速器，获取每批次数据
    for x,y in dataloader:
        # 5.1.切换测试模式
        model.eval()
        # 5.2.模型预测
        y_pred = model(x)
        # 5.3.argmax()函数功能：返回最大值对应的索引，充当→该图片的预测分类
        y_pred =torch.argmax(y_pred,dim=-1)  # -1表示行
        # 5.4.统计预测正确的样本个数
        total_correct += (y_pred == y).sum()
        # 5.5.统计总样本个数
        total_samples += len(y)
    # 6.打印正确率（预测结果）
    print(f'Acc:{total_correct / total_samples:.2f}')
# 5. 测试.
if __name__ == '__main__':
    # 1. 获取数据集.
    train_dataset, test_dataset = create_dataset()
    print(f'训练集: {train_dataset.data.shape}')       # (50000, 32, 32, 3)
    print(f'测试集: {test_dataset.data.shape}')         # (10000, 32, 32, 3)
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    print(f'数据集类别: {train_dataset.class_to_idx}')

    # 图像展示.
    plt.figure(figsize=(2, 2))
    plt.imshow(train_dataset.data[1111])       # 索引为1111的图像
    plt.title(train_dataset.targets[1111])
    plt.show()

    # 2.搭建神经网络.
    model = ImageModel()
    # 参1：模型，参2：输入维度（CHW，通道，高，宽），参3：批次大小
    summary(model,(3,32,32),BATCH_SIZE)

    # 3.模型训练.
    train(train_dataset)

    # 4.模型测试.
    evaluate(test_dataset)
    
