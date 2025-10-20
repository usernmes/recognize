import os

base_dir = "D:/深度大作业/"

# 创建所需目录
os.makedirs(os.path.join(base_dir, 'data/train/images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'data/train/labels'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'data/val/images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'data/val/labels'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)

print("项目目录结构创建完成！")