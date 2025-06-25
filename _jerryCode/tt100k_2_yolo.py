import os
import json
import shutil
import random
from tqdm import tqdm
from PIL import Image

import os
import json
import shutil
import random
from tqdm import tqdm
from PIL import Image

# ========== 配置参数 ==========
# 项目根目录配置 - 可根据实际情况修改
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# 输入路径配置
IMAGE_ROOT = os.path.join(PROJECT_ROOT, 'data')
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, 'data/annotations.json')

# 输出路径配置
OUTPUT_DATASET_NAME = 'yolo_dataset_filtered_20'
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, OUTPUT_DATASET_NAME)
YOLO_IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'images')
YOLO_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels')

# 创建输出目录
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'val'), exist_ok=True)

print(f"🏠 项目根目录: {PROJECT_ROOT}")
print(f"📂 输入数据目录: {IMAGE_ROOT}")
print(f"📄 标注文件: {ANNOTATION_FILE}")
print(f"📁 输出数据集目录: {OUTPUT_ROOT}")
print("-" * 60)

# ========== 加载标注 ==========
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

# 指定要转换的类别（按标注数量排序的前20类）
selected_categories = [
    'pn', 'pne', 'i5', 'p11', 'pl40', 'po', 'pl50', 'pl80', 'io', 'pl60',
    'p26', 'i4', 'pl100', 'pl30', 'pl5', 'il60', 'i2', 'p5', 'w57', 'p10'
]

category2id = {name: idx for idx, name in enumerate(selected_categories)}
print(f"选择转换 {len(selected_categories)} 类：{selected_categories}")

# 保存类别映射文件到输出目录
names_file_path = os.path.join(OUTPUT_ROOT, 'tt100k.names')
with open(names_file_path, "w") as f:
    for name in selected_categories:
        f.write(name + "\n")

# 创建 YOLO 配置文件
yaml_content = f"""# YOLO dataset configuration
# Dataset path
path: {OUTPUT_ROOT}

# Train and validation sets
train: images/train
val: images/val

# Number of classes
nc: {len(selected_categories)}

# Class names
names: {selected_categories}
"""

yaml_file_path = os.path.join(OUTPUT_ROOT, 'tt100k.yaml')
with open(yaml_file_path, "w", encoding='utf-8') as f:
    f.write(yaml_content)

# ========== 开始转换 ==========
# 筛选包含指定类别标注的图片
filtered_img_items = []
for img_id, img_info in data['imgs'].items():
    # 检查该图片是否包含指定类别的标注
    valid_objects = []
    for obj in img_info['objects']:
        if obj['category'] in selected_categories:
            valid_objects.append(obj)
    
    # 只有包含指定类别标注的图片才加入转换列表
    if valid_objects:
        img_info_copy = img_info.copy()
        img_info_copy['objects'] = valid_objects
        filtered_img_items.append((img_id, img_info_copy))

print(f"筛选出包含指定类别标注的图片数量: {len(filtered_img_items)}")

random.shuffle(filtered_img_items)
train_ratio = 0.8
split_index = int(len(filtered_img_items) * train_ratio)

for i, (img_id, img_info) in enumerate(tqdm(filtered_img_items)):
    phase = 'train' if i < split_index else 'val'
    src_img_path = os.path.join(IMAGE_ROOT, img_info['path'])
    if not os.path.exists(src_img_path):
        continue

    # 复制图像
    dst_img_path = os.path.join(YOLO_IMAGE_DIR, phase, os.path.basename(src_img_path))
    shutil.copyfile(src_img_path, dst_img_path)

    # 加载图像尺寸
    with Image.open(src_img_path) as img:
        width, height = img.size

    # 写入 YOLO 标注
    label_path = os.path.join(YOLO_LABEL_DIR, phase, os.path.splitext(os.path.basename(src_img_path))[0] + '.txt')
    with open(label_path, 'w') as f:
        for obj in img_info['objects']:
            cls = category2id[obj['category']]
            bbox = obj['bbox']
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ 转换完成！YOLO格式数据保存完毕")
print(f"📁 数据集路径: {OUTPUT_ROOT}")
print(f"📋 类别文件: {names_file_path}")
print(f"⚙️  配置文件: {yaml_file_path}")
print(f"🎯 训练集图片数: {split_index}")
print(f"🎯 验证集图片数: {len(filtered_img_items) - split_index}")
print(f"📊 总类别数: {len(selected_categories)}")
