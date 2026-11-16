import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import random
import numpy as np
import re
import matplotlib.pyplot as plt

# 配置
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 1. 数据集准备
class SiameseDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.people = {}  # {name: [path1, path2, ...]}
        
        # 扫描图片并按人名分类
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    # 尝试从文件名提取人名，例如: "乔宇轩无笑容1.jpg" -> "乔宇轩"
                    # 或者 "乔思倩-有笑容-1.jpg" -> "乔思倩"
                    name = re.split(r'[-有无笑脸脸容]', f)[0]
                    if name not in self.people:
                        self.people[name] = []
                    self.people[name].append(os.path.join(root, f))
        
        self.names = list(self.people.keys())
        self.num_people = len(self.names)
        print(f"找到 {self.num_people} 个不同的人脸类别。")

    def __len__(self):
        # 让每个 epoch 的样本数固定
        return 2000 

    def __getitem__(self, idx):
        # 50% 概率生成正样本对 (1)，50% 概率生成负样本对 (0)
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # 正样本: 同一个人
            name = random.choice(self.names)
            while len(self.people[name]) < 2: # 确保该人有多张图
                name = random.choice(self.names)
            img_path1 = random.choice(self.people[name])
            img_path2 = random.choice(self.people[name])
            while img_path1 == img_path2:
                img_path2 = random.choice(self.people[name])
            label = 1.0
        else:
            # 负样本: 不同的人
            name1 = random.choice(self.names)
            name2 = random.choice(self.names)
            while name1 == name2:
                name2 = random.choice(self.names)
            img_path1 = random.choice(self.people[name1])
            img_path2 = random.choice(self.people[name2])
            label = 0.0

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

# 2. 搭建孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 使用预训练的 ResNet18 作为主干
        self.resnet = models.resnet18(pretrained=True)
        # 去掉最后的全连接层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 最后的判别层: 输入为两个 512 维特征的差值
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1) # 展平为 512
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # 计算特征的绝对差值
        diff = torch.abs(output1 - output2)
        # 通过 FC 层输出相似度概率
        prediction = self.fc(diff)
        return prediction

# 3. 训练脚本
def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = SiameseDataset(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录训练数据用于绘图
    history = {'loss': [], 'acc': []}

    print(f"开始训练阶段一: 孪生匹配网络 (设备: {DEVICE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            preds = (output > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")

        # 记录每轮的平均值
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

    # 保存模型
    save_path = 'siamese_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"阶段一训练完成！模型已保存至: {save_path}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title('Siamese Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Accuracy')
    plt.title('Siamese Training Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plot_path = 'siamese_training_results.png'
    plt.savefig(plot_path)
    print(f"训练曲线已保存至: {plot_path}")

if __name__ == "__main__":
    train()
