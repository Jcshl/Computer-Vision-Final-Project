from ultralytics import YOLO


def train_model():
    # 加载预训练模型（可选yolov8n/s/m/l/x）
    model = YOLO("yolov8n.pt")  # 从头训练则用YOLO("yolov8n.yaml")

    # 训练参数配置
    results = model.train(
        data="data.yaml.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮次
        batch=16,  # 批次大小
        imgsz=640,  # 图像尺寸
        patience=50,  # 早停轮次
        device="0",  # 使用GPU，如 "0,1" 或 "cpu"
        workers=8,  # 数据加载线程数
        optimizer="auto",  # 优化器（auto/SGD/Adam）
        lr0=0.01,  # 初始学习率
        pretrained=True  # 是否加载预训练权重
    )

    # 训练后自动验证（可选）
    model.val()


if __name__ == "__main__":
    train_model()
