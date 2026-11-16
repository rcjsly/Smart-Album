import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models import resnet18
import matplotlib.pyplot as plt

BATCH_SIZE = 128
LR = 0.01
EPOCH = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_train = os.path.join(BASE_DIR, 'FER', 'resnet_train_set')
path_valid = os.path.join(BASE_DIR, 'FER', 'resnet_valid_set')

transforms_train = transforms.Compose([
    transforms.Grayscale(),  # 使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转 ±10 度
    transforms.RandomCrop(48, padding=4),  # 随机裁剪，假设输入尺寸为 48x48
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
    transforms.ToTensor()
])

transforms_valid = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

data_train = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
data_valid = torchvision.datasets.ImageFolder(root=path_valid, transform=transforms_valid)

train_set = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
valid_set = torch.utils.data.DataLoader(dataset=data_valid, batch_size=BATCH_SIZE, shuffle=False)


# 加载预定义的 ResNet18 模型
model = resnet18(pretrained=False)
# 修改输入通道为 1（灰度图）
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# 修改输出类别为 7（7种情绪）
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 添加 Dropout，丢弃率 50%
    nn.Linear(model.fc.in_features, 7)
)
model.to(DEVICE)
# 添加 L2 正则化 (weight_decay=0.0001)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
# optimizer.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 添加学习率调度器 - 每 15 个 epoch 衰减为原来的 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
y_pred = []

# 早停参数
best_valid_acc = 0.0
patience = 10  # 容忍验证准确率不提升的 epoch 数
counter = 0  # 计数器


def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    print(f"开始训练第 {epoch} 个epoch...")
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdims=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc.append(correct / len(data_train))
    train_loss.append(total_loss / len(dataset))
    print(f"Epoch {epoch} 完成，训练准确率: {100 * correct / len(data_train):.2f}%, 平均Loss: {total_loss / len(dataset):.4f}")


def valid(model, device, dataset):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            pred = output.max(1, keepdims=True)[1]
            global y_pred
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.item()

    valid_acc.append(correct / len(data_valid))
    valid_loss.append(total_loss / len(dataset))
    print(f"验证完成，验证准确率: {100 * correct / len(data_valid):.2f}%, 平均Loss: {total_loss / len(dataset):.4f}")


def RUN():
    global best_valid_acc, counter
    for epoch in range(1, EPOCH + 1):
        # 尝试动态学习率
        train(model, device=DEVICE, dataset=train_set, optimizer=optimizer, epoch=epoch)
        valid(model, device=DEVICE, dataset=valid_set)
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        current_valid_acc = valid_acc[-1]
        if current_valid_acc > best_valid_acc:
            best_valid_acc = current_valid_acc
            counter = 0
            # 保存最佳模型
            save_dir = os.path.join(BASE_DIR, 'FER', 'dataset')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model, os.path.join(save_dir, 'model_resnet.pkl'))
            print(f"验证准确率提升，保存最佳模型: {best_valid_acc:.4f}")
        else:
            counter += 1
            print(f"验证准确率未提升，计数: {counter}/{patience}")
            if counter >= patience:
                print(f"早停触发！最佳验证准确率: {best_valid_acc:.4f}")
                break


def plot_results():
    """绘制训练和验证的loss和准确率图表"""
    # 根据实际训练的 epoch 数生成 x 轴
    actual_epochs = range(1, len(train_loss) + 1)

    # 绘制loss图表
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(actual_epochs, train_loss, label='Training Loss')
    plt.plot(actual_epochs, valid_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率图表
    plt.subplot(1, 2, 2)
    plt.plot(actual_epochs, train_acc, label='Training Accuracy')
    plt.plot(actual_epochs, valid_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == '__main__':
    RUN()
    plot_results()
    print(f"训练完成！共训练 {len(train_loss)} 个 epoch")
    print(f"最佳验证准确率: {best_valid_acc:.4f}")

# 训练后得到模型
