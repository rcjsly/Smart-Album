# 模型文件

此仓库不附带本地模型权重。完整运行时，从你自己的可信模型备份恢复以下文件：

|路径|用途|
|---|---|
|detection/yolo11n.pt|YOLO检测；类别语义由实际权重决定|
|recognition/siamese_model.pth|孪生网络人脸匹配|
|recognition/metric_model.pth|度量学习人脸检索|
|segmentation/unet_face.pth|UNet分割实验|
|emotion/model_resnet.pkl|表情分类|

前端或训练脚本的部分YOLO配置使用项目根目录 `yolo11n.pt`，需按实际选择的配置放置。没有本地Haar XML时，后端会尝试OpenCV内置的人脸检测器。

标准YOLO目标检测权重的person类别不等同人脸检测模型。应根据任务使用对应权重，不能把通用模型当成人脸模型声称精度。权重与数据许可需由提供方确认。仅加载可信的PyTorch/pickle模型文件。
