import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from unet_model import UNet

# 配置
IMG_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\images'
MASK_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\masks'
MODEL_SAVE_PATH = r'c:\Users\14649\Desktop\智能相册\models\segmentation\unet_face.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50 # 增加训练轮数以提高精度
BATCH_SIZE = 4 # 降低 Batch 以防显存不足
LR = 0.0005
IMG_SIZE = 224 # 提高分辨率

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx].replace('.jpg', '.png'))
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
            # mask 只需要 Resize 和 ToTensor，不需要 Normalize
            mask_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
            
        return img, mask

def train():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SegmentationDataset(IMG_DIR, MASK_DIR, transform=transform)
    
    # 移除 100 张限制，使用全部数据进行训练
    if len(dataset) == 0:
        print("错误: 找不到训练数据，请先运行 auto_annotate.py")
        return
    print(f"找到 {len(dataset)} 张训练图片。")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"开始正式训练 UNet 人脸分割模型 (设备: {DEVICE})...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Current Loss: {loss.item():.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] 完成 | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} ---")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** 验证集 Loss 创历史新低，模型已保存至: {MODEL_SAVE_PATH} ***")
            
    print("\n训练全部结束！你可以重新启动 smile_api.py 来查看更精准的美白效果。")

if __name__ == '__main__':
    train()
