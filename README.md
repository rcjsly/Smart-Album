# 智能相册系统

基于 **Flask、PyTorch、OpenCV、MediaPipe 和 SQLite** 的本地智能相册项目，前端采用 HTML/CSS/JavaScript，集成照片管理、摄像头交互、人像编辑、人脸匹配与检索，以及可选的多模态文案与视频生成功能。

## 功能与代码

|模块|实现方式|
|---|---|
|相册管理|照片上传、浏览、删除；SQLite保存文件信息与特征|
|笑脸抓拍|表情推理与前端摄像头抓拍交互|
|人像编辑|MediaPipe关键点、OpenCV图像变形、背景模糊与图像修补|
|人脸匹配与检索|ResNet18特征、孪生网络、度量学习和余弦相似度|
|手势交互|MediaPipe Hands，前端交互及后端手势接口|
|AI文案|OpenAI兼容SDK调用阿里云Qwen多模态模型，需要自行配置密钥|
|视频生成|MoviePy图片剪辑、缩放与转场|

源码包含上述模块；功能可用性依赖本地模型、运行环境和数据。仓库未提供新的精度、帧率或线上效果验证。

## 本地运行

建议Python 3.10或3.11，在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# 可选：仅在使用AI文案时设置自己的阿里云密钥
$env:QWEN_API_KEY="在本机填入你自己的密钥"
python smile_api.py
```

浏览器打开 **http://localhost:5000**，允许摄像头权限。前端接口当前指向本机5000端口。AI文案功能未配置密钥时返回明确的503提示，其他模块不需要此密钥。

`.env.example`仅为配置示例，程序不会自动读取`.env`文件；请在运行后端的同一终端设置环境变量。不要把真实密钥提交到Git。

完整模型功能需要恢复[模型文件](models/README.md)。首次运行会初始化本地数据库与上传目录；照片与特征保存在本机，不包含在仓库中。视频生成还需要可用的FFmpeg运行环境。部分前端手势脚本来自jsDelivr CDN，需要网络。

原始依赖清单已补充后端实际导入的OpenAI SDK和MoviePy 1.x。PyTorch、torchvision及facenet-pytorch有版本兼容要求，GPU版本请按本机CUDA环境选择；本次上传没有安装并验证完整依赖组合。

## 目录

```text
smile_api.py              Flask接口、模型加载、图像处理与数据库操作
smart-album-system/      首页、相册、拍照、编辑与设置页面
scripts/training/        YOLO、表情和人脸模型训练脚本
scripts/segmentation/    UNet与标注实验
scripts/tools/           特征提取、数据准备、标注和导入工具
legacy/                  历史数据处理脚本
models/README.md         本地权重位置说明
docs/                    原项目开发知识库
```

训练脚本需要自行提供数据集，部分脚本保留原开发机绝对路径，运行前须修改。知识库是原开发笔记，包含方法与规划，不能作为已完成实验或性能验收的证明。

## 上传范围与验证

公开源码、页面和说明；本地照片、相册数据库、人脸特征、训练数据、训练输出、缓存及权重未上传。没有替换或删除原项目文件。

上传前已检查Python语法和常见密钥格式，并验证AI密钥缺失时的配置与接口返回逻辑。没有运行摄像头、模型推理、云端调用、图像编辑或完整系统验收。当前是本地开发应用，不包含账户认证和生产部署配置。

第三方依赖及模型遵循各自许可；本仓库未额外授予第三方模型或数据的使用权。
