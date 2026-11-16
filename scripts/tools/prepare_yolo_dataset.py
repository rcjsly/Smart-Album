import os
import shutil
import random
from glob import glob

def prepare_dataset():
    # 1. 配置路径
    src_dir = r'c:\Users\14649\Desktop\智能相册\dataset'
    target_dir = r'c:\Users\14649\Desktop\智能相册\yolo_dataset'
    
    # 定义目标结构
    subdirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    
    # 2. 创建目录结构
    for s in subdirs:
        os.makedirs(os.path.join(target_dir, s), exist_ok=True)
        
    # 3. 收集所有图片和对应的标注文件
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    pairs = []
    
    for ext in image_exts:
        images = glob(os.path.join(src_dir, '**', ext), recursive=True)
        for img_path in images:
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                pairs.append((img_path, txt_path))
    
    if not pairs:
        print(f"错误: 在 {src_dir} 中未找到已标注的图片对。请先运行标注工具。")
        return

    print(f"找到 {len(pairs)} 组标注好的图片。")
    
    # 4. 随机打乱并划分数据集 (80% 训练, 20% 验证)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # 5. 复制文件
    def copy_files(file_pairs, mode):
        for img_p, txt_p in file_pairs:
            # 获取文件名
            img_name = os.path.basename(img_p)
            txt_name = os.path.basename(txt_p)
            
            # 复制图片
            shutil.copy2(img_p, os.path.join(target_dir, 'images', mode, img_name))
            # 复制标注
            shutil.copy2(txt_p, os.path.join(target_dir, 'labels', mode, txt_name))
            
    print("正在复制训练集文件...")
    copy_files(train_pairs, 'train')
    print("正在复制验证集文件...")
    copy_files(val_pairs, 'val')
    
    print(f"\n数据集准备完成！")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"验证集: {len(val_pairs)} 张图片")
    print(f"目标目录: {target_dir}")

if __name__ == '__main__':
    prepare_dataset()
