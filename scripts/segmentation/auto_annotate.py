import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import tempfile
import shutil

# 配置
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset'
OUTPUT_IMG_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\images'
OUTPUT_MASK_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\masks'
CASCADE_PATH = r'c:\Users\14649\Desktop\智能相册\models\detection\haarcascade_frontalface_default.xml'

# 初始化人脸检测器 (使用临时路径解决中文路径加载失败)
def load_cascade(path):
    face_cascade = cv2.CascadeClassifier()
    if os.path.exists(path):
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tf:
            temp_path = tf.name
        shutil.copy2(path, temp_path)
        face_cascade.load(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return face_cascade

face_cascade = load_cascade(CASCADE_PATH)

def auto_segment(img_path):
    # 读取图片（支持中文路径）
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None:
        return None, None
    
    # 如果图片太大，先等比缩小以加速分割
    max_size = 512
    h_orig, w_orig = img.shape[:2]
    if max(h_orig, w_orig) > max_size:
        scale = max_size / max(h_orig, w_orig)
        img_small = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        scale = 1.0
        img_small = img.copy()

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None, None
    
    # 选最大的脸
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    
    # 使用 GrabCut 进行自动分割
    mask = np.zeros(img_small.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 增加区域，确保包含全脸及部分脖子（向下多扩一点）
    pad_w = int(w * 0.2)
    pad_h_top = int(h * 0.15)
    pad_h_bottom = int(h * 0.3) # 向下多扩一点，包含下巴和脖子
    rect = (max(0, x - pad_w), max(0, y - pad_h_top), min(img_small.shape[1], w + 2*pad_w), min(img_small.shape[0], h + pad_h_top + pad_h_bottom))
    
    try:
        # 增加迭代次数以提高标注质量
        cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 0 和 2 是背景，1 和 3 是前景
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        
        # 形态学操作：平滑掩码边缘并去除杂点
        kernel = np.ones((5,5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel) # 去噪
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0) # 平滑边缘
        _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
        
        # 将 mask 还原到原图尺寸
        if scale != 1.0:
            mask2 = cv2.resize(mask2, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
        return img, mask2
    except Exception as e:
        print(f"分割失败 {img_path}: {e}")
        return None, None

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(glob(os.path.join(DATASET_DIR, '**', ext), recursive=True))
    
    print(f"找到 {len(image_paths)} 张图片，开始自动标注...")
    
    count = 0
    for i, path in enumerate(tqdm(image_paths)):
        img, mask = auto_segment(path)
        
        if img is not None and mask is not None:
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            new_filename = f"face_{i:04d}.jpg"
            
            # 保存原图和掩码（支持中文路径）
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(OUTPUT_IMG_DIR, new_filename))
            cv2.imencode('.png', mask)[1].tofile(os.path.join(OUTPUT_MASK_DIR, new_filename.replace('.jpg', '.png')))
            count += 1
            
    print(f"标注完成！成功生成 {count} 组分割数据。")

if __name__ == '__main__':
    main()
