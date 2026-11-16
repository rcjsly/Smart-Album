import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sqlite3
import pickle
import numpy as np

# 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE = os.path.join(BASE_DIR, 'smart_album.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models/recognition/metric_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# 1. 加载特征提取模型
class MetricNetwork(nn.Module):
    def __init__(self):
        super(MetricNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x, p=2, dim=1) # L2 归一化，使得余弦距离 = 欧氏距离
        return x

def get_model():
    model = MetricNetwork().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"成功加载模型: {MODEL_PATH}")
    else:
        print(f"警告: 找不到模型 {MODEL_PATH}，将使用初始权重。")
    return model

# 2. 特征提取函数
def extract_feature(img_path, model, transform):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feature = model(img_tensor).cpu().numpy().flatten()
        return feature
    except Exception as e:
        print(f"提取特征失败 {img_path}: {e}")
        return None

# 3. 预提取所有图片的特征
def pre_extract_all():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = get_model()
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # 确保数据库中有 embedding 列
    try:
        cursor.execute("ALTER TABLE photos ADD COLUMN embedding BLOB")
    except sqlite3.OperationalError:
        pass # 列已存在

    cursor.execute("SELECT id, path FROM photos")
    rows = cursor.fetchall()

    print(f"开始为 {len(rows)} 张照片提取特征...")
    
    for photo_id, img_path in rows:
        if not os.path.exists(img_path):
            continue
            
        feature = extract_feature(img_path, model, transform)
        if feature is not None:
            # 使用 pickle 将特征向量序列化并存入数据库
            cursor.execute("UPDATE photos SET embedding = ? WHERE id = ?", (pickle.dumps(feature), photo_id))
    
    conn.commit()
    conn.close()
    print("所有照片特征预提取完成！")

if __name__ == "__main__":
    pre_extract_all()
