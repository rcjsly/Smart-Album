import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import random
import re
import matplotlib.pyplot as plt

# 配置
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MARGIN = 2.0 # 增大 Margin 增加训练难度

# 1. 三元组数据集准备
class TripletDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.people = {}  # {name: [path1, path2, ...]}
        
        # 扫描并分类
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    name = re.split(r'[-有无笑脸脸容]', f)[0]
                    if name not in self.people:
                        self.people[name] = []
                    self.people[name].append(os.path.join(root, f))
        
        self.names = [n for n in self.people.keys() if len(self.people[n]) >= 2]
        self.all_names = list(self.people.keys())
        print(f"找到 {len(self.names)} 个可用于训练的三元组类别。")

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        # Anchor (A)
        name_a = random.choice(self.names)
        anchor_path = random.choice(self.people[name_a])
        
        # Positive (P)
        pos_path = random.choice(self.people[name_a])
        while pos_path == anchor_path:
            pos_path = random.choice(self.people[name_a])
            
        # Negative (N)
        name_n = random.choice(self.all_names)
        while name_n == name_a:
            name_n = random.choice(self.all_names)
        neg_path = random.choice(self.people[name_n])
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

# 2. 搭建特征提取网络
class MetricNetwork(nn.Module):
    def __init__(self):
        super(MetricNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # 去掉最后分类层，输出 512 维特征
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1) # 512
        # L2 归一化，使得特征向量在单位球面上，从而点积 = 余弦相似度
        x = F.normalize(x, p=2, dim=1)
        return x

# 3. 训练脚本
def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TripletDataset(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MetricNetwork().to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录训练数据用于绘图
    history = {'loss': []}

    print(f"开始训练阶段二: 度量学习特征提取网络 (设备: {DEVICE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (anchor, pos, neg) in enumerate(dataloader):
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            anchor_feat = model(anchor)
            pos_feat = model(pos)
            neg_feat = model(neg)
            
            loss = criterion(anchor_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 计算平均 Loss
        avg_loss = running_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"--- Epoch {epoch+1} 结束, 平均 Loss: {avg_loss:.4f} ---")
        
        # 保存每轮的模型 (防止按 Ctrl+C 丢失进度)
        save_path = 'metric_model.pth'
        torch.save(model.state_dict(), save_path)
        print(f"模型已自动保存至: {save_path}")
        
        # 如果平均 Loss 已经非常接近 0，则提前停止训练
        if avg_loss < 0.0001:
            print("Loss 已降至接近 0，模型已收敛，自动提前结束训练！")
            break

    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Loss')
    plt.title('Metric Learning Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_path = 'metric_training_results.png'
    plt.savefig(plot_path)
    print(f"训练曲线已保存至: {plot_path}")

    print(f"训练流程结束。")

if __name__ == "__main__":
    train()
