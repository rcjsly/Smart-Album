import os
import requests
import base64
import glob
from tqdm import tqdm

# 配置
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset_cropped'
API_URL = 'http://localhost:5000/upload_photo'

# 获取所有图片
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png']
image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(glob.glob(os.path.join(DATASET_DIR, '**', ext), recursive=True))

print(f"找到 {len(image_paths)} 张图片，准备导入相册...")

for img_path in tqdm(image_paths):
    try:
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        data = {
            'image': f"data:image/jpeg;base64,{img_base64}"
        }
        
        response = requests.post(API_URL, json=data)
        if response.status_code != 200:
            print(f"导入失败 {img_path}: {response.text}")
            
    except Exception as e:
        print(f"处理出错 {img_path}: {e}")

print("导入完成！")
