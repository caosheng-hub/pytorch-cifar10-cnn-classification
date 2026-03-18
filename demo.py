import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# ===================== 配置参数 =====================
BATCH_SIZE = 8
EPOCHS = 3  # 简化训练轮数，加快演示速度
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./model/image_model.pth"


# ===================== 1. 数据集准备 =====================
def create_dataset():
    """创建CIFAR10训练集和测试集"""
    # 自动下载CIFAR10数据集（约170MB）
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {train_dataset.class_to_idx}")
    return train_dataset, test_dataset


# ===================== 2. 定义卷积神经网络 =====================
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 + 池化层
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)  # 输入3通道，输出6通道，3*3卷积核
        self.pool1 = nn.MaxPool2d(2, 2, 0)  # 2*2最大池化
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)  # 输入6通道，输出16通道
        self.pool2 = nn.MaxPool2d(2, 2, 0)  # 2*2最大池化

        # 全连接层
        self.linear1 = nn.Linear(16 * 6 * 6, 120)  # 展平后输入维度16*6*6=576
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)  # 10分类输出

    def forward(self, x):
        # 卷积层1 + ReLU + 池化
        x = self.pool1(torch.relu(self.conv1(x)))
        # 卷积层2 + ReLU + 池化
        x = self.pool2(torch.relu(self.conv2(x)))
        # 展平：(batch, 16, 6, 6) → (batch, 576)
        x = x.reshape(x.size(0), -1)
        # 全连接层1 + ReLU
        x = torch.relu(self.linear1(x))
        # 全连接层2 + ReLU
        x = torch.relu(self.linear2(x))
        # 输出层（无激活，后续用CrossEntropyLoss）
        return self.output(x)


# ===================== 3. 模型训练 =====================
def train_model(train_dataset):
    """训练模型并保存"""
    # 创建数据加载器（打乱数据）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 初始化模型、损失函数、优化器
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失（内置Softmax）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 创建模型保存目录
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 开始训练
    print("\n===== 开始训练 =====")
    for epoch in range(EPOCHS):
        total_loss, total_correct, total_samples = 0.0, 0, 0
        start_time = time.time()

        # 遍历每个批次
        for x, y in train_loader:
            model.train()  # 训练模式（启用Dropout/BatchNorm等）
            y_pred = model(x)  # 前向传播
            loss = criterion(y_pred, y)  # 计算损失

            # 反向传播 + 优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 统计指标
            total_loss += loss.item() * len(y)  # 累计损失
            total_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()  # 累计正确数
            total_samples += len(y)  # 累计样本数

        # 打印本轮训练结果
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        epoch_time = time.time() - start_time
        print(
            f"轮次 {epoch + 1}/{EPOCHS} | 平均损失: {epoch_loss:.4f} | 准确率: {epoch_acc:.4f} | 耗时: {epoch_time:.2f}秒")

    # 保存训练好的模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")
    return model


# ===================== 4. 模型测试 =====================
def test_model(test_dataset):
    """加载模型并测试准确率"""
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()  # 测试模式（禁用Dropout/BatchNorm等）

    print("\n===== 开始测试 =====")
    total_correct, total_samples = 0, 0
    with torch.no_grad():  # 禁用梯度计算（加快速度，节省内存）
        for x, y in test_loader:
            y_pred = model(x)
            y_pred_label = torch.argmax(y_pred, dim=1)  # 获取预测类别
            total_correct += (y_pred_label == y).sum().item()
            total_samples += len(y)

    test_acc = total_correct / total_samples
    print(f"测试集准确率: {test_acc:.4f}")
    return test_acc


# ===================== 主函数 =====================
if __name__ == '__main__':
    # 1. 加载数据集
    train_dataset, test_dataset = create_dataset()

    # 2. 训练模型
    train_model(train_dataset)

    # 3. 测试模型
    test_model(test_dataset)