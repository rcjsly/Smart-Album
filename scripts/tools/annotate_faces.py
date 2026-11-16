import cv2
import os
import glob
import numpy as np
import torch
try:
    from facenet_pytorch import MTCNN
    has_mtcnn = True
except ImportError:
    has_mtcnn = False
    print("警告: 未安装 facenet-pytorch，无法使用 MTCNN。请运行: pip install facenet-pytorch")

# 配置
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset'
MAX_DISPLAY_DIM = 1000 # 最大显示高度或宽度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化检测器
mtcnn = None
face_cascade = None

if has_mtcnn:
    try:
        mtcnn = MTCNN(keep_all=True, device=DEVICE)
        print(f"成功加载 MTCNN 检测器 (运行设备: {DEVICE})")
    except Exception as e:
        print(f"MTCNN 加载失败: {e}")
        has_mtcnn = False

if not has_mtcnn:
    # 退回到 Haar Cascade
    CASCADE_PATH = os.path.join(os.path.dirname(__file__), '../../models/detection/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier()
    if os.path.exists(CASCADE_PATH):
        face_cascade.load(CASCADE_PATH)
    if face_cascade.empty():
        builtin_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade.load(builtin_path)
    print("使用 Haar Cascade 检测器")

# 获取所有图片
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png']
image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(glob.glob(os.path.join(DATASET_DIR, '**', ext), recursive=True))

if not image_paths:
    print(f"在 {DATASET_DIR} 中未找到图片。")
    exit()

current_idx = 0
drawing = False
ix, iy = -1, -1
boxes = []

def save_yolo_labels(img_path, boxes, img_w, img_h):
    txt_path = os.path.splitext(img_path)[0] + '.txt'
    with open(txt_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # 转换为 YOLO 格式: class_id x_center y_center width height (归一化)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            xc = (x1 + x2) / 2.0 / img_w
            yc = (y1 + y2) / 2.0 / img_h
            nw = w / img_w
            nh = h / img_h
            f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
    print(f"保存标注: {txt_path}")

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, boxes, img_copy
    scale = param['scale']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_copy.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Annotation Tool', img_temp)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        # 将显示坐标映射回原图坐标
        orig_ix, orig_iy = int(ix / scale), int(iy / scale)
        orig_x, orig_y = int(x / scale), int(y / scale)
        boxes.append((orig_ix, orig_iy, orig_x, orig_y))
        cv2.imshow('Annotation Tool', img_copy)

cv2.namedWindow('Annotation Tool')
# 初始调用 setMouseCallback，之后在循环中根据图片更新参数
cv2.setMouseCallback('Annotation Tool', draw_rect, param={'scale': 1.0})

print("\n--- 人脸标注工具操作指南 ---")
print("1. 鼠标左键拖动: 绘制矩形框")
print("2. 's' 键: 保存当前标注并进入下一张")
print("3. 'c' 键: 清除当前所有框 (包括自动标注)")
print("4. 'q' 键: 退出工具")
print("5. 'n' 键: 跳过当前图片")
print("提示: 紫色框为自动检测结果，红色为已有标注")
print("---------------------------\n")

while current_idx < len(image_paths):
    img_path = image_paths[current_idx]
    # 使用 numpy 加载并解码，解决 Windows 下中文路径读取问题
    try:
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图片出错 {img_path}: {e}")
        img = None
        
    if img is None:
        print(f"无法解码图片: {img_path}")
        current_idx += 1
        continue
        
    img_h, img_w = img.shape[:2]
    # 计算缩放比例
    scale = 1.0
    if img_h > MAX_DISPLAY_DIM or img_w > MAX_DISPLAY_DIM:
        scale = MAX_DISPLAY_DIM / max(img_h, img_w)
    
    display_h, display_w = int(img_h * scale), int(img_w * scale)
    img_display = cv2.resize(img, (display_w, display_h))
    img_copy = img_display.copy()
    # 更新回调函数的参数
    cv2.setMouseCallback('Annotation Tool', draw_rect, param={'scale': scale})
    
    boxes = []
    
    # 尝试加载已有标注
    txt_path = os.path.splitext(img_path)[0] + '.txt'
    has_existing_labels = False
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, nw, nh = map(float, parts)
                    x1 = int((xc - nw/2) * img_w)
                    y1 = int((yc - nh/2) * img_h)
                    x2 = int((xc + nw/2) * img_w)
                    y2 = int((yc + nh/2) * img_h)
                    boxes.append((x1, y1, x2, y2))
                    # 绘制时映射回显示坐标
                    cv2.rectangle(img_copy, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (255, 0, 0), 2)
                    has_existing_labels = True
    
    # 如果没有已有标注，尝试自动检测
    if not has_existing_labels:
        if mtcnn is not None:
            # 使用 MTCNN 检测 (更准确)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected_boxes, _ = mtcnn.detect(img_rgb)
            if detected_boxes is not None:
                for box in detected_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(img_copy, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (128, 0, 128), 2)
                print(f"MTCNN 自动检测到 {len(detected_boxes)} 张人脸")
        elif face_cascade is not None:
            # 使用 Haar Cascade (作为备选)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in detected_faces:
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append((x1, y1, x2, y2))
                cv2.rectangle(img_copy, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (128, 0, 128), 2)
            if len(detected_faces) > 0:
                print(f"Haar Cascade 自动检测到 {len(detected_faces)} 张人脸")

    while True:
        cv2.imshow('Annotation Tool', img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # Save
            save_yolo_labels(img_path, boxes, img_w, img_h)
            current_idx += 1
            break
        elif key == ord('c'): # Clear
            img_copy = img_display.copy()
            boxes = []
        elif key == ord('n'): # Next (skip)
            current_idx += 1
            break
        elif key == ord('q'): # Quit
            current_idx = len(image_paths)
            break

cv2.destroyAllWindows()
