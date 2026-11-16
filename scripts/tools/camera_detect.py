import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
try:
    from facenet_pytorch import MTCNN
    has_mtcnn = True
except ImportError:
    has_mtcnn = False

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 尝试加载您训练的最佳 YOLO 模型
YOLO_MODEL_PATH = 'runs/detect/yolo_face_model/weights/best.pt'
if not os.path.exists(YOLO_MODEL_PATH):
    # 如果还没训练完，使用预训练模型作为演示
    YOLO_MODEL_PATH = 'yolo11n.pt'

print(f"正在加载模型: {YOLO_MODEL_PATH} ...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    detection_mode = "YOLO"
except Exception as e:
    print(f"YOLO 加载失败: {e}，将尝试使用 MTCNN")
    if has_mtcnn:
        model = MTCNN(keep_all=True, device=DEVICE)
        detection_mode = "MTCNN"
    else:
        # 最终退回到 Haar Cascade
        model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detection_mode = "HAAR"

def start_camera():
    cap = cv2.VideoCapture(0) # 打开默认摄像头
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    print(f"摄像头已启动！当前模式: {detection_mode}")
    print("按 'q' 键退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像翻转，更符合自拍习惯
        frame = cv2.flip(frame, 1)
        
        # 执行检测
        if detection_mode == "YOLO":
            results = model(frame, verbose=False, conf=0.5)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    # 绘制矩形框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        elif detection_mode == "MTCNN":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = model.detect(img_rgb)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"MTCNN {prob:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        else: # HAAR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = model.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示结果
        cv2.imshow('Real-time Face Detection', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
