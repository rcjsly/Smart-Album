import cv2
import os
import numpy as np
from glob import glob

# 尝试加载 MediaPipe
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    has_mediapipe = True
    print("成功加载 MediaPipe Face Mesh")
except Exception as e:
    has_mediapipe = False
    print(f"MediaPipe 加载失败: {e}")

# 总是加载 YOLO 作为备选或主检测器
try:
    from ultralytics import YOLO
    YOLO_MODEL_PATH = r'c:\Users\14649\Desktop\智能相册\models\detection\yolo11n.pt'
    yolo_detector = YOLO(YOLO_MODEL_PATH)
    has_yolo = True
    print("成功加载 YOLO 检测器")
except Exception as e:
    has_yolo = False
    print(f"YOLO 加载失败: {e}")

# 获取人脸边缘的关键点索引 (MediaPipe 468 关键点中的边缘点)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# --- 状态变量 ---
drawing = False # 是否正在涂抹
mode = True # True 为前景(白色)，False 为背景(黑色)
ix, iy = -1, -1
brush_size = 15

# --- 工具函数 ---
def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mode, brush_size
    mask_to_edit = param['mask']

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            color = 255 if mode else 0
            cv2.circle(mask_to_edit, (x, y), brush_size, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# --- 配置 ---
DATASET_DIR = r'c:\Users\14649\Desktop\智能相册\dataset'
OUTPUT_IMG_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\images'
OUTPUT_MASK_DIR = r'c:\Users\14649\Desktop\智能相册\data\segmentation\masks'

def get_auto_mask(img):
    """自适应检测器：优先使用 MediaPipe，失败则使用 YOLO 椭圆策略"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    
    if has_mediapipe:
        # --- MediaPipe 极致精准多边形策略 ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            hull_points = []
            for idx in FACE_OVAL:
                pt = face_landmarks.landmark[idx]
                hull_points.append([int(pt.x * w), int(pt.y * h)])
            hull_points = np.array(hull_points, dtype=np.int32)
            cv2.fillPoly(mask, [hull_points], 255)
            
            # 形态学平滑
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask

    # --- YOLO 椭圆引导策略 (MediaPipe 失败或未安装时的回退方案) ---
    if has_yolo:
        results = yolo_detector(img, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = sorted(results[0].boxes, key=lambda b: b.conf[0], reverse=True)[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            fw, fh = x2 - x1, y2 - y1
            
            gc_mask = np.zeros((h, w), np.uint8)
            rx, ry = max(0, x1), max(0, y1)
            rw, rh = min(w - rx, fw), min(h - ry, fh)
            gc_mask[ry:ry+rh, rx:rx+rw] = cv2.GC_PR_BGD
            
            center = (x1 + fw // 2, y1 + fh // 2)
            axes = (int(fw * 0.45), int(fh * 0.55))
            cv2.ellipse(gc_mask, center, axes, 0, 0, 360, cv2.GC_PR_FGD, -1)
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
                mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                return mask
            except:
                pass

    # 最终兜底：中心圆形
    cv2.circle(mask, (w//2, h//2), min(w, h)//3, 255, -1)
    return mask

def main():
    global mode, brush_size
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(glob(os.path.join(DATASET_DIR, '**', ext), recursive=True))

    print(f"找到 {len(image_paths)} 张图片。")
    print("\n--- 快捷键说明 ---")
    print("'s': 保存当前并进入下一张")
    print("'n': 跳过当前（不保存）")
    print("'f': 切换为【画前景】(白色)")
    print("'b': 切换为【画背景】(黑色)")
    print("'+': 增大画笔尺寸")
    print("'-': 减小画笔尺寸")
    print("'r': 重置为自动标注结果")
    print("'c': 清空所有标注（从零开始手动标）")
    print("'q': 退出程序")

    cv2.namedWindow('Interactive Annotation')
    
    for i, path in enumerate(image_paths):
        # 读取原图
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if img is None: continue
        
        # 统一缩放到合适高度以便标注
        h, w = img.shape[:2]
        display_h = 700
        scale = display_h / h
        img_small = cv2.resize(img, (int(w*scale), display_h))
        
        # 初始自动标注
        current_mask = get_auto_mask(img_small)
        auto_mask_backup = current_mask.copy()
        
        param = {'mask': current_mask, 'orig': img_small}
        cv2.setMouseCallback('Interactive Annotation', draw_mask, param)

        while True:
            # 叠加显示：原图 + 半透明红色掩码
            overlay = img_small.copy()
            overlay[current_mask == 255] = [0, 0, 255] # 红色区域代表选中
            display_img = cv2.addWeighted(img_small, 0.7, overlay, 0.3, 0)
            
            # 显示状态
            status = f"Mode: {'Foreground' if mode else 'Background'} | Brush: {brush_size} | [{i+1}/{len(image_paths)}]"
            cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Interactive Annotation', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # Save
                # 还原 mask 到原图尺寸并保存
                final_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                new_filename = f"face_{i:04d}"
                cv2.imencode('.jpg', img)[1].tofile(os.path.join(OUTPUT_IMG_DIR, new_filename + ".jpg"))
                cv2.imencode('.png', final_mask)[1].tofile(os.path.join(OUTPUT_MASK_DIR, new_filename + ".png"))
                print(f"已保存: {new_filename}")
                break
            elif key == ord('n'): # Next
                break
            elif key == ord('f'): mode = True
            elif key == ord('b'): mode = False
            elif key == ord('+') or key == ord('='): brush_size = min(brush_size + 5, 100)
            elif key == ord('-'): brush_size = max(brush_size - 5, 5)
            elif key == ord('r'): current_mask[:] = auto_mask_backup[:]
            elif key == ord('c'): current_mask[:] = 0 # 清空掩码
            elif key == ord('q'): 
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("标注流程结束。")

if __name__ == '__main__':
    main()
