import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
from tqdm import tqdm

# 解决 OpenMP 运行时库冲突问题 (OMP: Error #15)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 解决图像文件损坏/不完整的问题 (OSError: image file is truncated)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. 数据预处理与增强 (Preprocessing & Augmentation)
# 增加数据增强强度以防止过拟合
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.RandomRotation(15),        # 增加旋转角度到15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 随机颜色抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(train_dir='FER/resnet_train_set', val_dir='FER/resnet_valid_set', num_epochs=15, batch_size=64, learning_rate=0.0001):
    # 2. 数据集加载
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"错误: 找不到训练或验证数据集目录。")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    
    # 动态获取类别数量
    num_classes = len(train_dataset.classes)
    print(f"检测到类别数量: {num_classes}")
    print(f"类别名称: {train_dataset.classes}")

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. 构建多分类模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes) # 修改为 FER 数据集的类别数 (7)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 4. 训练循环
    for epoch in range(num_epochs):
        # ... 保持之前的训练循环逻辑 ...
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 使用 tqdm 显示训练进度
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in pbar:
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条信息
                pbar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"警告: 训练跳过损坏图片: {e}")
                continue

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            # 使用 tqdm 显示验证进度
            vbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in vbar:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
                    
                    # 更新进度条信息
                    vbar.set_postfix({'val_loss': loss.item()})
                except Exception as e:
                    print(f"警告: 验证跳过损坏图片: {e}")
                    continue

        val_loss = val_running_loss / val_size
        val_acc = val_running_corrects.double() / val_size
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # 5. 可视化结果 (Visualization)
    plot_results(history)
    
    # 保存模型
    torch.save(model.state_dict(), 'smile_detector_model.pth')
    print("模型已保存为 'smile_detector_model.pth'")

def plot_results(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("训练统计图已保存为 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    # 使用 FER 数据集进行 7 类表情识别训练
    train_model(train_dir='FER/resnet_train_set', val_dir='FER/resnet_valid_set', num_epochs=15, batch_size=64, learning_rate=0.0001)
