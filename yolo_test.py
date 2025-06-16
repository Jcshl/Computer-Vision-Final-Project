from ultralytics import YOLO
import torch
import cv2

# 1. 加载训练好的模型
model = YOLO("best.pt")  # 确保best.pt在当前目录下

# 2. 进行视频检测
results = model.predict(
    source="Chess_video_example.mp4",  # 视频路径
    save=True,              # 保存检测结果视频
    conf=0.5,               # 置信度阈值（可调）
    device="cuda" if torch.cuda.is_available() else "cpu",  # 自动选择GPU/CPU
    show=True,              # 实时显示检测画面（如果不需要可关闭）
    imgsz=640               # 输入图像大小（根据训练设置调整）
)

# 3. 提示结果位置
print(f"[提示] 检测完成！输出视频保存在：{results[0].save_dir}")
