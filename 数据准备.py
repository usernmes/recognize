import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# 基础路径
base_dir = "D:/深度大作业/"
data_dir = os.path.join(base_dir, "data")

# 加载 FER2013 数据集
csv_path = os.path.join(base_dir, 'fer2013.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"未找到数据集文件: {csv_path}")
data = pd.read_csv(csv_path)

# 检查数据格式
if 'emotion' not in data.columns or 'pixels' not in data.columns:
    raise ValueError("CSV 文件缺少必要的列: 'emotion' 或 'pixels'")


# 数据处理函数
def process_data(row, mode='train'):
    try:
        # 转换像素字符串为数组并重塑为图像
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        label = row['emotion']

        # 文件路径
        image_path = os.path.join(data_dir, f"{mode}/images/{row.name}.jpg")
        label_path = os.path.join(data_dir, f"{mode}/labels/{row.name}.txt")

        # 创建所需目录
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        # 使用 PIL 保存图片 (确保图片为 RGB 格式)
        image = Image.fromarray(pixels)
        image_rgb = image.convert("RGB")  # 转换为 RGB 格式

        # 保存图片
        image_rgb.save(image_path, format="JPEG")

        # 保存标签 (YOLO 格式)
        h, w = 48, 48  # 图像宽高
        cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0  # 中心点坐标和宽高比例
        with open(label_path, 'w') as f:
            f.write(f"{label} {cx} {cy} {bw} {bh}\n")

        return True
    except Exception as e:
        print(f"处理数据时出错: {row.name}, 错误信息: {e}")
        return False


# 分割数据为训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 处理训练数据
print("正在处理训练数据...")
for _, row in tqdm(train_data.iterrows(), total=train_data.shape[0]):
    process_data(row, mode='train')

# 处理验证数据
print("正在处理验证数据...")
for _, row in tqdm(val_data.iterrows(), total=val_data.shape[0]):
    process_data(row, mode='val')

print("数据处理完成！")
