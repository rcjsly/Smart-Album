import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import sqlite3
from datetime import datetime
import uuid
import cv2
import numpy as np
import tempfile
import shutil
import pickle
import torch.nn.functional as F
from ultralytics import YOLO
from openai import OpenAI
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
# 导入 UNet 模型
from scripts.segmentation.unet_model import UNet
import mediapipe as mp

# 尝试从 facenet-pytorch 导入 MTCNN 用于检测 (作为备选)
try:
    from facenet_pytorch import MTCNN
    has_mtcnn = True
except ImportError:
    has_mtcnn = False
    print("警告: 未安装 facenet-pytorch，检测功能将退回到 Haar Cascade")

# 解决 OpenMP 运行时库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 全局配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'smart_album.db'
SIAMESE_PATH = 'models/recognition/siamese_model.pth'
METRIC_PATH = 'models/recognition/metric_model.pth'
UNET_PATH = 'models/segmentation/unet_face.pth'

# 通义千问 Qwen-VL 配置
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "").strip()
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) if QWEN_API_KEY else None

app = Flask(__name__)
CORS(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 预处理人脸识别图片 (224x224)
face_rec_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 1.1 孪生匹配网络 (阶段一)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 显式使用 weights 参数避免 UserWarning
        self.resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        diff = torch.abs(output1 - output2)
        return self.fc(diff)

# 1.2 度量学习特征网络 (阶段二)
class MetricNetwork(nn.Module):
    def __init__(self):
        super(MetricNetwork, self).__init__()
        # 显式使用 weights 参数避免 UserWarning
        self.resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        return F.normalize(x, p=2, dim=1)

# 1.3 初始化模型
siamese_model = None
metric_model = None
unet_model = None

# 初始化 MediaPipe Pose 用于瘦腿
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True, # 改为 True，确保每一帧都能独立精准识别人体
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 初始化 MediaPipe Face Mesh 用于高精度人脸分割（取代 UNet）
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 初始化 MediaPipe Hands 用于手势识别
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True, # 对于单张请求，True 更稳健
    max_num_hands=1,
    min_detection_confidence=0.5, # 稍微调低置信度门槛，更容易识别到手
    min_tracking_confidence=0.5
)

# 初始化 MediaPipe Selfie Segmentation 用于精准人像分割
mp_selfie = mp.solutions.selfie_segmentation
selfie_segmenter = mp_selfie.SelfieSegmentation(model_selection=0) # 0 为通用模型

def load_face_recognition_models():
    global siamese_model, metric_model, unet_model
    try:
        if os.path.exists(SIAMESE_PATH):
            siamese_model = SiameseNetwork().to(DEVICE)
            siamese_model.load_state_dict(torch.load(SIAMESE_PATH, map_location=DEVICE))
            siamese_model.eval()
            print(f"成功加载孪生匹配模型: {SIAMESE_PATH}")
        
        if os.path.exists(METRIC_PATH):
            metric_model = MetricNetwork().to(DEVICE)
            metric_model.load_state_dict(torch.load(METRIC_PATH, map_location=DEVICE))
            metric_model.eval()
            print(f"成功加载度度量检索模型: {METRIC_PATH}")
            
        if os.path.exists(UNET_PATH):
            unet_model = UNet().to(DEVICE)
            unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
            unet_model.eval()
            print(f"成功加载人脸分割模型: {UNET_PATH}")
    except Exception as e:
        print(f"人脸识别模型加载失败: {e}")

load_face_recognition_models()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. 自动加载人脸检测器 (解决 Windows 中文路径乱码问题)
original_cascade_path = os.path.join(os.path.dirname(__file__), 'models/detection/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier()

if os.path.exists(original_cascade_path):
    try:
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tf:
            temp_path = tf.name
        shutil.copy2(original_cascade_path, temp_path)
        face_cascade.load(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if not face_cascade.empty():
            print(f"成功通过中转路径加载根目录检测器: {original_cascade_path}")
    except Exception as e:
        print(f"中转加载失败: {e}")

if face_cascade.empty():
    builtin_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade.load(builtin_path)

# 1.1 初始化 YOLO 人脸检测模型
yolo_detector = None
current_detection_mode = "haar" # 可选 "haar", "yolo"

def load_yolo_model(model_path='models/detection/yolo11n.pt'):
    global yolo_detector
    try:
        yolo_detector = YOLO(model_path)
        print(f"成功加载 YOLO 检测模型: {model_path}")
        return True
    except Exception as e:
        print(f"加载 YOLO 模型失败: {e}")
        return False

# 启动时加载 YOLO 模型
load_yolo_model()

# 2. 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            emotion TEXT,
            confidence REAL,
            timestamp TEXT NOT NULL,
            type TEXT DEFAULT 'photo',
            embedding BLOB
        )
    ''')
    # 检查并添加缺失列
    cols = [c[1] for c in cursor.execute('PRAGMA table_info(photos)').fetchall()]
    if 'emotion' not in cols:
        cursor.execute('ALTER TABLE photos ADD COLUMN emotion TEXT')
    if 'embedding' not in cols:
        cursor.execute('ALTER TABLE photos ADD COLUMN embedding BLOB')
    conn.commit()
    conn.close()

init_db()

# --- 人脸识别模型初始化 ---
# 使用阶段二的度量学习模型提取 512 维特征
def get_face_embedding(image):
    if metric_model is None:
        return None
    try:
        # 1. 首先进行人脸检测与裁剪
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # 稍微向外扩一点，保证特征完整
            padding = int(w * 0.1)
            face_img = image.crop((max(0, x-padding), max(0, y-padding), min(image.width, x+w+padding), min(image.height, y+h+padding)))
        else:
            # 如果没检测到脸，直接用原图尝试（虽然效果可能打折）
            face_img = image

        # 2. 提取特征
        img_tensor = face_rec_transform(face_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = metric_model(img_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"特征提取逻辑出错: {e}")
    return None

# --- 3. 表情模型加载 ---
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
current_model_info = {"name": "自训练模型", "path": "FER/dataset/model_resnet_best.pkl"}

def load_emotion_model(model_path):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 7))
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            model.eval()
            print(f"成功加载表情模型: {model_path}")
            return model
        except Exception as e:
            print(f"模型加载出错: {e}")
    return None

emotion_model = load_emotion_model(current_model_info["path"])

# 预处理逻辑
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# --- 表情与笑脸检测逻辑 ---
def analyze_face_smile(face_landmarks):
    """
    通过 Face Mesh 关键点几何关系计算笑脸强度
    """
    # 嘴部关键点索引 (MediaPipe 标准)
    # 61, 291: 左右嘴角
    # 0, 17: 上下唇中心
    lm = face_landmarks.landmark
    
    # 1. 计算嘴巴宽度
    mouth_width = np.sqrt((lm[61].x - lm[291].x)**2 + (lm[61].y - lm[291].y)**2)
    
    # 2. 计算人脸宽度（用于归一化）
    # 454, 234: 左右脸边缘
    face_width = np.sqrt((lm[454].x - lm[234].x)**2 + (lm[454].y - lm[234].y)**2)
    
    # 3. 计算嘴角上扬程度
    # 嘴角 y 坐标与上唇中心 y 坐标的相对关系
    mouth_avg_y = (lm[61].y + lm[291].y) / 2
    smile_up = lm[0].y - mouth_avg_y
    
    # 归一化得分
    ratio = mouth_width / face_width
    # 正常表情 ratio 约 0.35, 笑的时候会增加到 0.45+
    
    # 综合判定得分 (0.0 - 1.0)
    smile_score = np.clip((ratio - 0.35) / 0.15 + (smile_up * 10), 0, 1)
    
    return float(smile_score)

def perform_prediction(image):
    """
    重构后的表情预测：优先使用 MediaPipe 几何分析检测笑脸
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    results = face_mesh_detector.process(img_rgb)
    
    if results.multi_face_landmarks:
        # 使用高精度的几何分析
        smile_score = analyze_face_smile(results.multi_face_landmarks[0])
        
        if smile_score > 0.4:
            return "happy", smile_score
        else:
            return "neutral", 1.0 - smile_score
            
    return "unknown", 0.0

import sys

# --- 美白算法 ---
def apply_whitening(image, mask, whitening_strength=30):
    """
    基于分割 Mask 的人脸美白算法
    image: PIL Image
    mask: numpy array (0-255)
    whitening_strength: 美白强度 (0-100)
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. 皮肤平滑（双边滤波）
    smooth = cv2.bilateralFilter(img_cv, d=15, sigmaColor=75, sigmaSpace=75)
    
    # 2. 亮度提升 (美白) - 在 LAB 空间调整 L 分量
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    
    # 显著提升亮度以供测试
    # 映射强度：0-100 对应增加 0-80 亮度
    increase = int(whitening_strength * 0.8)
    l = cv2.add(l, increase)
    
    lab_whitened = cv2.merge((l, a, b_chan))
    whitened = cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)
    
    # 3. 混合：只在 mask 区域应用美白效果
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    mask_3ch = cv2.GaussianBlur(mask_3ch, (15, 15), 0)
    
    result = (whitened * mask_3ch + img_cv * (1 - mask_3ch)).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

# --- 全身瘦身算法 (全局水平拉伸) ---
def apply_slimming(image, strength=30):
    """
    全局瘦身：通过水平压缩图像实现变瘦效果
    image: PIL Image
    strength: 瘦身强度 (0-100)
    """
    if strength <= 0: return image
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    
    # 强度映射：0-100 对应 0.85 - 1.0 的水平缩放比例
    # 30 强度对应约 0.95 缩放
    factor = 1.0 - (strength / 100.0) * 0.15
    
    # 1. 计算新宽度
    new_w = int(w * factor)
    if new_w <= 0: return image
    
    # 2. 水平缩放
    resized = cv2.resize(img_cv, (new_w, h), interpolation=cv2.INTER_LINEAR)
    
    # 3. 填充回原尺寸，保持画面比例不拉伸（背景留黑边或拉伸回原宽）
    # 为了实现“变瘦”视觉感，我们直接拉伸回原宽，但这样会造成整体变瘦
    # 更好的做法是只对中心区域进行缩放，但用户要求简单拉伸，这里采用全局瘦身逻辑
    final = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

# --- 高级美颜算法 ---
def apply_advanced_beauty(img_cv, strength=50):
    """
    使用 MediaPipe Face Mesh 进行精准磨皮与美白
    """
    h, w = img_cv.shape[:2]
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return img_cv, False
        
    # 1. 创建高精度人脸掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    face_landmarks = results.multi_face_landmarks[0]
    
    # 获取面部轮廓索引 (MediaPipe 标准索引)
    # 这里的索引涵盖了整个脸部区域
    hull_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
        54, 103, 67, 109
    ]
    points = []
    for idx in hull_indices:
        lm = face_landmarks.landmark[idx]
        points.append([int(lm.x * w), int(lm.y * h)])
    
    cv2.fillConvexPoly(mask, np.array(points), 255)
    
    # 排除眼睛和嘴巴区域（可选，为了更自然）
    # 这里简单对掩码进行模糊处理以获得柔和边缘
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    # 2. 磨皮：使用双边滤波结合高通滤波（细节保留）
    # 磨皮强度
    smooth_strength = strength / 100.0
    dst = cv2.bilateralFilter(img_cv, d=0, sigmaColor=int(30 * smooth_strength), sigmaSpace=15)
    
    # 3. 美白：使用色彩曲线模拟，更自然地提升亮部
    # 强度 0-100 对应 1.0 - 1.3 的增益
    gain = 1.0 + (strength / 100.0) * 0.25
    whitened = cv2.multiply(dst, np.array([gain, gain, gain]))
    whitened = np.clip(whitened, 0, 255).astype(np.uint8)
    
    # 4. 混合
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (whitened * mask_3ch + img_cv * (1 - mask_3ch)).astype(np.uint8)
    
    return result, True

@app.route('/whiten_face', methods=['POST'])
def whiten_face():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        strength = int(data.get('strength', 50))
        
        # 使用高级美颜算法
        result, success = apply_advanced_beauty(img_cv, strength)
        
        # 4. 返回结果
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success',
            'detected': success
        })
    except Exception as e:
        print(f"!!! 美白处理出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500
        
        # 如果整张图都没有掩码，尝试给一个兜底的中心区域掩码
        if mask_binary.max() == 0:
            print("DEBUG: UNet 未检测到人脸，使用中心区域兜底", flush=True)
            h, w = mask_binary.shape
            cv2.circle(mask_binary, (w//2, h//2), min(w, h)//3, 255, -1)
        else:
            print("DEBUG: 成功提取到 AI 分割掩码", flush=True)

        mask_resized = cv2.resize(mask_binary, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
        
        # 3. 应用美白
        strength = max(30, int(data.get('strength', 30)))
        result_img = apply_whitening(img, mask_resized, strength)
        
        # 4. 返回图片
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print(">>> 美白处理成功，已返回结果 <<<", flush=True)
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}"
        })
    except Exception as e:
        print(f"!!! 美白处理出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/slim_legs', methods=['POST'])
def slim_legs():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img_cv.shape[:2]
        
        strength = int(data.get('strength', 50))
        # 强度映射：0.15 为最大压缩度
        distort_factor = (strength / 100.0) * 0.15
        
        # 1. 使用与编辑页面一致的 Pose 检测方案
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            return jsonify({'image': data['image'], 'status': 'no_person_detected'})
            
        lm = results.pose_landmarks.landmark
        
        # 2. 计算人像中轴线 (基于肩膀和胯部)
        # 这里的关键是确定“哪里需要变瘦”
        left_shoulder_x = lm[11].x * w
        right_shoulder_x = lm[12].x * w
        left_hip_x = lm[23].x * w
        right_hip_x = lm[24].x * w
        
        # 躯干中心 X
        body_center_x = (left_shoulder_x + right_shoulder_x + left_hip_x + right_hip_x) / 4
        # 躯干估算宽度 (用于限定变形范围)
        body_width = abs(left_shoulder_x - right_shoulder_x) * 1.5
        
        # 3. 构造重映射表
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        
        # 4. 局部“线性捏合”算法 (只对身体中轴线附近的区域起作用)
        # 定义变形范围：身体中心左右 1.2 倍身宽的区域
        start_x = max(0, int(body_center_x - body_width))
        end_x = min(w, int(body_center_x + body_width))
        
        # 预计算变形偏移
        x_indices = np.arange(start_x, end_x)
        offsets = x_indices - body_center_x
        
        # 使用余弦衰减函数，确保背景（边缘）位移平滑回归 0
        # 这样身体中心压缩最厉害，背景直线完全不弯曲
        weights = np.cos(np.clip(offsets / body_width * (np.pi / 2), -np.pi/2, np.pi/2))
        
        # 应用变形
        map_x[:, start_x:end_x] = body_center_x + offsets / (1 - distort_factor * weights)
        
        # 5. 执行重映射
        final = cv2.remap(img_cv, map_x, map_y, cv2.INTER_LINEAR)
        
        _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success',
            'detected': True
        })
    except Exception as e:
        print(f"!!! 实时瘦身处理出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/leg_slimming', methods=['POST'])
def leg_slimming():
    """
    高级瘦腿算法：基于 Pose 关键点的局部挤压
    只对腿部区域进行横向收缩，保护上半身和背景
    """
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img_cv.shape[:2]
        
        strength = int(data.get('strength', 50))
        # 强度映射：最大收缩 20%
        squeeze_factor = (strength / 100.0) * 0.2
        
        # 1. 检测姿态
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            return jsonify({'image': data['image'], 'status': 'no_person_detected'})
            
        lm = results.pose_landmarks.landmark
        
        # 2. 获取腿部关键坐标
        # 胯部 (23, 24), 膝盖 (25, 26), 脚踝 (27, 28)
        hip_y = int((lm[23].y + lm[24].y) / 2 * h)
        ankle_y = int((lm[27].y + lm[28].y) / 2 * h)
        
        # 左右腿中心线
        left_leg_x = (lm[23].x + lm[25].x + lm[27].x) / 3 * w
        right_leg_x = (lm[24].x + lm[26].x + lm[28].x) / 3 * w
        
        # 3. 构造重映射表
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        
        # 4. 分别对左右腿应用挤压
        # 设定挤压的垂直范围：从胯部略上到脚踝
        y_start = max(0, hip_y - int(h * 0.05))
        y_end = min(h, ankle_y + int(h * 0.05))
        
        # 定义挤压的影响宽度（根据两腿间距动态调整）
        influence_width = abs(right_leg_x - left_leg_x) * 0.6
        
        for leg_x in [left_leg_x, right_leg_x]:
            # 只在腿部垂直范围内操作
            for y in range(y_start, y_end):
                # 垂直方向的渐变权重：靠近胯部和脚踝的地方挤压变小，中间（膝盖附近）最强
                v_dist = abs(y - (hip_y + ankle_y)/2)
                v_max_dist = (y_end - y_start) / 2
                v_weight = np.cos(np.clip(v_dist / v_max_dist * (np.pi / 2), 0, np.pi/2))
                
                # 水平方向的挤压范围
                h_start = max(0, int(leg_x - influence_width))
                h_end = min(w, int(leg_x + influence_width))
                
                # 核心挤压逻辑
                x_indices = np.arange(h_start, h_end)
                offsets = x_indices - leg_x
                
                # 水平权重：靠近腿中心线挤压最强，背景边缘为 0
                h_weights = np.cos(np.clip(offsets / influence_width * (np.pi / 2), -np.pi/2, np.pi/2))
                
                # 更新映射表
                map_x[y, h_start:h_end] = leg_x + offsets / (1 - squeeze_factor * v_weight * h_weights)

        # 5. 执行重映射
        final = cv2.remap(img_cv, map_x, map_y, cv2.INTER_LINEAR)
        
        _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success',
            'detected': True
        })
    except Exception as e:
        print(f"!!! 高级瘦腿处理出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/blur_background', methods=['POST'])
def blur_background():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img_cv.shape[:2]

        # 1. 使用 Face Mesh 获取人脸掩码（作为人像的核心区域）
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(img_rgb)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # 1. 扩充关键点范围，包含额头、发际线以及更宽的面部边缘
            # 这里的索引涵盖了面部外轮廓以及发际线附近的参考点
            hull_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
                400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
                54, 103, 67, 109,
                # 额外增加头顶上方的虚拟点，防止头发被切掉
                10, 151, 337, 299, 333, 298, 301, 70, 63, 105, 66, 107
            ]
            points = []
            for idx in hull_indices:
                lm = face_landmarks.landmark[idx]
                points.append([int(lm.x * w), int(lm.y * h)])
            
            # 2. 向上和向外微调点，确保头发被包含
            poly_points = np.array(points)
            # 计算中心点
            center_x = np.mean(poly_points[:, 0])
            center_y = np.mean(poly_points[:, 1])
            
            # 将点向外推一点点 (1.1倍)，特别是头顶方向
            for p in poly_points:
                dx = p[0] - center_x
                dy = p[1] - center_y
                p[0] = int(center_x + dx * 1.05)
                # 对头顶上方的点（dy < 0）推得更远一点，以包含头发
                p[1] = int(center_y + dy * (1.15 if dy < 0 else 1.05))

            cv2.fillConvexPoly(mask, poly_points, 255)
            
            # 3. 动态计算肩膀区域
            x_min = poly_points[:, 0].min()
            x_max = poly_points[:, 0].max()
            y_max = poly_points[:, 1].max()
            # 肩膀比脸宽
            shoulder_width = (x_max - x_min) * 0.8
            cv2.rectangle(mask, (int(x_min - shoulder_width), int(y_max - 20)), (int(x_max + shoulder_width), h), 255, -1)
        
        # 4. 形态学操作：膨胀，让掩码稍微外扩，确保边缘细节（如头发丝）不被误伤
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 5. 平滑掩码边缘
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        # 2. 生成背景模糊图
        background_blur = cv2.GaussianBlur(img_cv, (51, 51), 0)
        
        # 3. 混合
        result = (img_cv * mask_3ch + background_blur * (1 - mask_3ch)).astype(np.uint8)
        
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success'
        })
    except Exception as e:
        print(f"!!! 背景羽化出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

# --- 路人自动擦除接口 ---
@app.route('/remove_passerby', methods=['POST'])
def remove_passerby():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img_cv.shape[:2]

        # 1. 使用 YOLO 检测所有人
        results = yolo_detector(img_cv, verbose=False, classes=[0])
        
        # 2. 识别路人（非中心且较小的检测框）
        mask = np.zeros((h, w), dtype=np.uint8)
        img_center_x = w / 2
        
        passerby_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                
                # 如果检测框不在中心区域（例如偏离中心 20% 以上）
                if abs(cx - img_center_x) > (w * 0.15):
                    # 标记为路人区域
                    cv2.rectangle(mask, (x1-10, y1-10), (x2+10, y2+10), 255, -1)
                    passerby_count += 1
        
        # 3. 使用 Inpainting 擦除路人
        if passerby_count > 0:
            # 简单的图像修补算法
            result = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
        else:
            result = img_cv

        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success',
            'removed_count': passerby_count
        })
    except Exception as e:
        print(f"!!! 路人擦除出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

# --- AI 场景识别与文案生成接口 ---
@app.route('/generate_ai_copy', methods=['POST'])
def generate_ai_copy():
    if qwen_client is None:
        return jsonify({"error": "AI文案功能未配置：请设置 QWEN_API_KEY 环境变量"}), 503
    try:
        data = request.json
        image_url = data.get('image') # Base64 数据
        
        if not image_url:
            return jsonify({'error': '未提供图片数据'}), 400

        # 调用 Qwen-VL-Plus 模型
        completion = qwen_client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url # OpenAI 兼容模式支持直接传 Base64
                            },
                        },
                        {
                            "type": "text", 
                            "text": "你是一个资深自媒体博主。请识别这张照片的场景（如：海边、餐厅、办公室），分析照片的氛围和情感，并以此生成 3 个不同风格（如：励志感、幽默感、文艺范）的小红书风格文案，包含合适的 Emoji 和标签。"
                        },
                    ],
                },
            ],
        )
        
        answer = completion.choices[0].message.content
        
        return jsonify({
            'success': True,
            'copy': answer
        })
    except Exception as e:
        print(f"!!! AI 文案生成出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

# --- 腿部拉长算法 ---
@app.route('/leg_stretch', methods=['POST'])
def leg_stretch():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img_cv.shape[:2]
        
        strength = int(data.get('strength', 50))
        # 强度映射：0-100 对应 1.0 - 1.2 的拉伸比例
        stretch_factor = 1.0 + (strength / 100.0) * 0.2
        
        # 1. 使用 MediaPipe Pose 检测人体姿态
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            print("DEBUG: 未检测到人体，执行全局底部拉长", flush=True)
            # 如果没检测到人，对底部 40% 区域进行拉长
            y_start = int(h * 0.6)
            roi = img_cv[y_start:h, :]
            roi_h = roi.shape[0]
            new_roi_h = int(roi_h * stretch_factor)
            stretched_roi = cv2.resize(roi, (w, new_roi_h), interpolation=cv2.INTER_LINEAR)
            
            # 合成回原图（保持原高度，需要裁剪或整体缩放，这里选择裁剪底部以保持比例）
            final = np.zeros_like(img_cv)
            final[0:y_start, :] = img_cv[0:y_start, :]
            # 填充拉长后的部分，由于高度限制，只取前部分
            fill_h = h - y_start
            final[y_start:h, :] = stretched_roi[0:fill_h, :]
            
            success = False
        else:
            print("DEBUG: 检测到人体，开始精准腿部拉长", flush=True)
            lm = results.pose_landmarks.landmark
            # 获取胯部(23,24)和脚踝(27,28)的平均 Y 坐标
            hip_y = int((lm[23].y + lm[24].y) / 2 * h)
            ankle_y = int((lm[27].y + lm[28].y) / 2 * h)
            
            if ankle_y <= hip_y: # 异常情况处理
                return jsonify({'image': data['image'], 'status': 'no_legs_detected'})

            # 定义拉长区域：从胯部到脚踝
            leg_roi = img_cv[hip_y:ankle_y, :]
            leg_h = leg_roi.shape[0]
            new_leg_h = int(leg_h * stretch_factor)
            
            # 拉伸腿部区域
            stretched_leg = cv2.resize(leg_roi, (w, new_leg_h), interpolation=cv2.INTER_LINEAR)
            
            # 计算总高度变化
            diff = new_leg_h - leg_h
            
            # 创建新画布（高度增加以容纳拉长后的腿）
            # 或者保持原高度，压缩上方/下方区域。这里采用保持原高度并裁剪脚部以下的策略，这是最自然的
            final = np.zeros_like(img_cv)
            # 1. 复制上半身
            final[0:hip_y, :] = img_cv[0:hip_y, :]
            # 2. 填充拉长后的腿
            fill_leg_h = min(new_leg_h, h - hip_y)
            final[hip_y:hip_y+fill_leg_h, :] = stretched_leg[0:fill_leg_h, :]
            # 3. 填充剩余部分（如果有）
            current_y = hip_y + fill_leg_h
            if current_y < h:
                remaining_h = h - current_y
                source_y = ankle_y
                if source_y + remaining_h <= h:
                    final[current_y:h, :] = img_cv[source_y:source_y+remaining_h, :]
            
            success = True

        _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({
            'image': f"data:image/jpeg;base64,{img_str}",
            'status': 'success',
            'detected': success
        })
    except Exception as e:
        print(f"!!! 腿部拉长出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

# --- 多图视频合成接口 ---
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        data = request.json
        photo_ids = data.get('photo_ids', [])
        duration_per_image = float(data.get('duration', 2.0)) # 每张图显示时间
        
        if not photo_ids:
            return jsonify({'error': '未选择图片'}), 400
            
        # 1. 获取图片路径
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(photo_ids))
        cursor.execute(f"SELECT filename FROM photos WHERE id IN ({placeholders})", photo_ids)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({'error': '未找到对应的图片文件'}), 404
            
        clips = []
        # 2. 创建视频片段（增加动态缩放效果）
        for row in rows:
            img_path = os.path.join(UPLOAD_FOLDER, row[0])
            if os.path.exists(img_path):
                # 基础剪辑：统一尺寸为 1280x720 (720p)
                clip = ImageClip(img_path).set_duration(duration_per_image)
                
                # 肯·伯恩斯效应 (Ken Burns Effect)：平滑动态缩放
                # 模拟从 1.0 倍逐渐放大到 1.15 倍，让静态照片“动”起来
                clip = clip.resize(lambda t: 1.0 + 0.15 * (t / duration_per_image))
                
                # 设置最终输出尺寸并居中
                clip = clip.set_position('center').resize(height=720)
                
                # 增加淡入淡出转场效果 (0.5秒)
                clip = clip.crossfadein(0.5).crossfadeout(0.5)
                
                clips.append(clip)
        
        if not clips:
            return jsonify({'error': '图片文件读取失败'}), 500
            
        # 3. 合成视频 (使用 padding 模式处理淡入淡出重叠)
        final_clip = concatenate_videoclips(clips, method="compose", padding=-0.5)
        
        # 4. 保存视频
        output_filename = f"album_video_{int(time.time())}.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # 使用 libx264 编码，保证兼容性
        final_clip.write_videofile(output_path, fps=24, codec="libx264", audio=False)
        
        # 5. 释放内存
        for c in clips: c.close()
        final_clip.close()
        
        return jsonify({
            'success': True,
            'video_url': f"http://localhost:5000/uploads/{output_filename}",
            'filename': output_filename
        })
    except Exception as e:
        print(f"!!! 视频合成出错: {e} !!!", flush=True)
        return jsonify({'error': str(e)}), 500

# --- 摄像头手势识别接口 ---
@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        
        gesture = "none"
        if not results.multi_hand_landmarks:
            # 增加没有检测到手的日志
            print("DEBUG: No hands detected in current frame", flush=True)
        else:
            print(f"DEBUG: Found {len(results.multi_hand_landmarks)} hand(s)", flush=True)
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            
            # 调试：打印关键点坐标（可选）
            # print(f"DEBUG: Index Finger Tip X: {lm[8].x:.2f}, Y: {lm[8].y:.2f}", flush=True)

            # 1. 判断张开/握拳 (用于缩放)
            # 计算食指尖(8)到掌心(0)的距离
            # 考虑到摄像头距离不同，使用中指根部(9)到掌心(0)作为参考长度
            ref_dist = np.sqrt((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2)
            dist = np.sqrt((lm[8].x - lm[0].x)**2 + (lm[8].y - lm[0].y)**2)
            
            # 相对比例判定
            ratio = dist / ref_dist if ref_dist > 0 else 0
            # print(f"DEBUG: Hand Ratio: {ratio:.2f}", flush=True)

            if ratio > 2.2: # 食指伸直，比例通常大于 2
                gesture = "open"
            elif ratio < 1.3: # 食指弯曲（握拳），比例通常小于 1.5
                gesture = "fist"
            
            # 2. 判断左右挥动 (用于切换)
            # 使用食指尖的水平位置，由于做了镜像，需要反转逻辑
            # index_x 在 0-1 之间，0 是左边缘，1 是右边缘
            index_x = lm[8].x
            if index_x < 0.25:
                gesture = "swipe_right" # 镜像后，手往左划（屏幕右边）对应切换上一张
            elif index_x > 0.75:
                gesture = "swipe_left"  # 镜像后，手往右划（屏幕左边）对应切换下一张
                
        if gesture != "none":
            print(f"DEBUG: Detected Gesture: {gesture}", flush=True)

        return jsonify({
            'gesture': gesture,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API 接口 ---

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        photo_id = str(uuid.uuid4())
        filename = f"{photo_id}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        emotion, confidence = perform_prediction(image)
        embedding = get_face_embedding(image)
        embedding_blob = pickle.dumps(embedding) if embedding is not None else None
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO photos (id, filename, path, emotion, confidence, timestamp, type, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (photo_id, filename, filepath, emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'photo', embedding_blob))
        conn.commit()
        conn.close()
        
        return jsonify({'id': photo_id, 'emotion': emotion, 'confidence': confidence, 'url': f'http://localhost:5000/uploads/{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_smile', methods=['POST'])
def predict_smile():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        emotion, confidence = perform_prediction(image)
        
        # 同时返回人脸框位置，方便前端绘制
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_rects = []
        if current_detection_mode == "yolo" and yolo_detector is not None:
            results = yolo_detector(cv_img, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_rects.append({'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1})
        else:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_rects.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        return jsonify({
            'emotion': emotion, 
            'confidence': confidence, 
            'faces': face_rects,
            'is_smiling': (emotion == 'happy' and confidence > 0.7)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/match_faces', methods=['POST'])
def match_faces():
    """阶段一：孪生网络匹配判断 (0/1)"""
    if siamese_model is None:
        return jsonify({'error': '孪生匹配模型未加载'}), 500
    try:
        data = request.json
        img1_data = data['image1'].split(',')[1]
        img2_data = data['image2'].split(',')[1]
        
        img1 = Image.open(io.BytesIO(base64.b64decode(img1_data))).convert('RGB')
        img2 = Image.open(io.BytesIO(base64.b64decode(img2_data))).convert('RGB')
        
        t1 = face_rec_transform(img1).unsqueeze(0).to(DEVICE)
        t2 = face_rec_transform(img2).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            similarity = siamese_model(t1, t2).item()
            
        return jsonify({
            'similarity': similarity,
            'is_same_person': similarity > 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_face', methods=['POST'])
def search_face():
    """阶段二：基于度量学习的特征检索 (余弦相似度)"""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        query_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 调试：检查模型是否加载
        if metric_model is None:
            return jsonify({'error': '检索模型(metric_model.pth)未加载，请检查 models/recognition/ 目录'}), 500

        query_embedding = get_face_embedding(query_img)
        if query_embedding is None:
            return jsonify({'error': '特征提取失败：可能未在图片中检测到清晰人脸'}), 400
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, embedding FROM photos WHERE embedding IS NOT NULL')
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            db_id, fname, db_emb_blob = row
            db_embedding = pickle.loads(db_emb_blob)
            
            # 计算余弦相似度 (由于已做 L2 归一化，点积即余弦相似度)
            similarity = np.dot(query_embedding, db_embedding)
            
            # 相似度阈值建议 0.6 以上
            if similarity > 0.5: 
                results.append({
                    'id': db_id,
                    'url': f'http://localhost:5000/uploads/{fname}',
                    'similarity': float(similarity)
                })
        
        conn.close()
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return jsonify(results[:10])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_photos', methods=['GET'])
def get_photos():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 核心修改：按 timestamp 降序排列，确保最新照片排在前面
        cursor.execute('SELECT * FROM photos ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        photos = [dict(row) for row in rows]
        for p in photos:
            p['url'] = f'http://localhost:5000/uploads/{p["filename"]}'
            p.pop('embedding', None) # 不返回二进制数据
        conn.close()
        return jsonify({'photos': photos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global emotion_model, current_model_info
    data = request.json
    model_type = data.get('type') # 'custom' or 'pretrained'
    
    if model_type == 'custom':
        path = "FER/dataset/model_resnet_best.pkl"
        name = "自训练表情模型"
    else:
        path = "models/emotion/model_resnet.pkl"
        name = "预训练表情模型"
        
    new_model = load_emotion_model(path)
    if new_model:
        emotion_model = new_model
        current_model_info = {"name": name, "path": path}
        return jsonify({'success': True, 'current_model': name})
    return jsonify({'error': '模型加载失败'}), 400

@app.route('/switch_detection', methods=['POST'])
def switch_detection():
    global current_detection_mode, yolo_detector
    data = request.json
    mode = data.get('mode') # 'haar' or 'yolo'
    model_path = data.get('path', 'yolo11n.pt') # 默认用预训练的
    
    if mode == 'yolo':
        if load_yolo_model(model_path):
            current_detection_mode = "yolo"
            return jsonify({'success': True, 'mode': 'YOLO', 'model': model_path})
        else:
            return jsonify({'error': 'YOLO 模型加载失败'}), 400
    else:
        current_detection_mode = "haar"
        return jsonify({'success': True, 'mode': 'Haar Cascade'})

@app.route('/delete_item/<id>', methods=['DELETE'])
def delete_item(id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT path FROM photos WHERE id = ?', (id,))
        row = cursor.fetchone()
        if row:
            if os.path.exists(row[0]): os.remove(row[0])
            cursor.execute('DELETE FROM photos WHERE id = ?', (id,))
            conn.commit()
            conn.close()
            return jsonify({'success': True})
        conn.close()
        return jsonify({'error': '找不到记录'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# 静态文件路由
@app.route('/')
def index():
    return send_from_directory('smart-album-system', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('smart-album-system', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
