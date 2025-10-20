import os
from ultralytics import YOLO

def main():
    # 设置训练参数
    base_dir = "D:/深度大作业/"
    yaml_path = os.path.join(base_dir, 'faces.yaml')

    # 加载 YOLOv8 模型
    model = YOLO('D:/深度大作业/yolov8x.pt')  # 加载预训练的基础模型

    results = model.train(
        amp=False,
        data=yaml_path,  # 数据集配置文件
        epochs=20,  # 训练轮数
        imgsz=128,  # 图片尺寸
        device=0,  # 使用 GPU 训练（0 为第一个 GPU，-1 为 CPU）
        save=True,  # 确保训练时保存模型
        batch=8,  # 降低批量大小
        cos_lr=True,  # 使用余弦学习率调度器
        lr0=0.005,  # 降低初始学习率
        lrf=0.1,  # 最终学习率
        weight_decay=0.0005,  # 添加权重衰减
        dropout=0.1,  # 使用 Dropout
        patience=10,  # 提前停止训练
        freeze=0,  # 解冻所有层
        project='D:/深度大作业/runs/train',
        name='face_expression_model'
    )

    print("训练完成！")

if __name__ == '__main__':
    main()
