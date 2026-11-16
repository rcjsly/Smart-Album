import cv2
import os
import numpy as np

# 1. 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def cv_imread(file_path):
    """
    使用 numpy 读取包含中文路径的图片
    """
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img
    except Exception as e:
        print(f"读取图片失败: {file_path}, 错误: {e}")
        return None

def cv_imwrite(file_path, img):
    """
    使用 numpy 保存包含中文路径的图片
    """
    try:
        cv2.imencode('.jpg', img)[1].tofile(file_path)
        return True
    except Exception as e:
        print(f"保存图片失败: {file_path}, 错误: {e}")
        return False

def crop_faces(input_base_dir='dataset', output_base_dir='dataset_cropped'):
    """
    遍历 input_base_dir 下的子文件夹，识别图片中的人脸并裁剪保存到 output_base_dir
    """
    # 定义要处理的类别
    categories = ['smiling', 'not_smiling']
    
    # 检查输入目录是否存在
    if not os.path.exists(input_base_dir):
        print(f"错误: 找不到输入目录 '{input_base_dir}'")
        return

    # 创建输出根目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"创建输出目录: {output_base_dir}")

    for category in categories:
        input_dir = os.path.join(input_base_dir, category)
        output_dir = os.path.join(output_base_dir, category)
        
        if not os.path.exists(input_dir):
            print(f"跳过类别 '{category}': 找不到文件夹 {input_dir}")
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"正在处理类别: {category}...")
        
        count = 0
        success_count = 0
        
        # 遍历文件夹中的图片
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(input_dir, filename)
                
                # 使用支持中文的读取方式
                img = cv_imread(img_path)
                if img is None:
                    continue
                
                # 转为灰度图，加速检测
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 2. 检测人脸
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # 3. 裁剪并保存
                for (x, y, w, h) in faces:
                    # 适当扩大人脸裁剪范围 (上下左右增加 10% 边缘)
                    padding_w = int(w * 0.1)
                    padding_h = int(h * 0.1)
                    
                    y1 = max(0, y - padding_h)
                    y2 = min(img.shape[0], y + h + padding_h)
                    x1 = max(0, x - padding_w)
                    x2 = min(img.shape[1], x + w + padding_w)
                    
                    face_img = img[y1:y2, x1:x2]
                    
                    # 统一缩放到 224x224
                    face_img = cv2.resize(face_img, (224, 224))
                    
                    # 使用支持中文的保存方式
                    output_path = os.path.join(output_dir, f"crop_{count}_{filename}")
                    if cv_imwrite(output_path, face_img):
                        success_count += 1
                        count += 1
                    
                    # 每张图只取第一个检测到的人脸
                    break 

        print(f"类别 '{category}' 处理完成: 成功裁剪 {success_count} 张人脸")

if __name__ == "__main__":
    # 执行批量裁剪
    # 默认从 'dataset' 读取，保存到 'dataset_cropped'
    crop_faces('dataset', 'dataset_cropped')
    print("\n所有处理已完成！请检查 'dataset_cropped' 文件夹。")
    print("提示: 建议使用裁剪后的数据集重新运行训练脚本以获得更高准确率。")
