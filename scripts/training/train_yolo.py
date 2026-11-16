from ultralytics import YOLO
import os
import torch

# 解决 OpenMP 运行时库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_yolo():
    # 1. 自动检测并选择最佳设备
    if torch.cuda.is_available():
        device = 0 # 使用第一个 GPU
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，将使用 GPU 训练。")
    else:
        device = 'cpu'
        print("未检测到可用 GPU，将使用 CPU 训练 (速度会较慢)。")

    # 2. 加载预训练模型
    model = YOLO('yolo11n.pt') 
    
    # 3. 检查数据集配置文件
    data_yaml = 'yolo_data.yaml'
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到 {data_yaml}")
        return

    # 4. 开始训练
    print(f"开始训练 YOLO 模型 (设备: {device})...")
    results = model.train(
        data=data_yaml,
        epochs=20,          # 缩短训练轮数为 20 轮，平衡时间与效果
        imgsz=640,          # 图片大小
        batch=8,            # 降低批次大小，减轻显存压力 (原为 16)
        workers=0,          # 设置为 0 以解决 Windows 页面文件不足 (WinError 1455)
        device=device,      # 自动切换 GPU/CPU
        name='yolo_face_model', 
        exist_ok=True,      # 允许在同一个文件夹运行
        resume=False,       # 因为修改了轮数，建议从头开始一个快速训练
        verbose=True,       
        plots=True,         
        save=True           
    )
    
    print(f"训练完成！模型保存在: {results.save_dir}")

if __name__ == '__main__':
    train_yolo()
