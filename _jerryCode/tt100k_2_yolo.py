import os
import json
import shutil
import random
from tqdm import tqdm
from PIL import Image

# ========== 设置路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT = os.path.join(BASE_DIR, '../../data')
ANNOTATION_FILE = os.path.join(BASE_DIR, '../../data/annotations.json')

YOLO_IMAGE_DIR = os.path.join(BASE_DIR, '../../yolo_dataset/images')
YOLO_LABEL_DIR = os.path.join(BASE_DIR, '../../yolo_dataset/labels')
os.makedirs(YOLO_IMAGE_DIR + '/train', exist_ok=True)
os.makedirs(YOLO_IMAGE_DIR + '/val', exist_ok=True)
os.makedirs(YOLO_LABEL_DIR + '/train', exist_ok=True)
os.makedirs(YOLO_LABEL_DIR + '/val', exist_ok=True)

# ========== 加载标注 ==========
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

# 统计所有类别
category_set = set()
for img_info in data['imgs'].values():
    for obj in img_info['objects']:
        category_set.add(obj['category'])

category_list = sorted(list(category_set))
category2id = {name: idx for idx, name in enumerate(category_list)}
print(f"共识别出 {len(category_list)} 类：{category_list}")

# 保存类别映射文件（方便写yaml用）
with open("tt100k.names", "w") as f:
    for name in category_list:
        f.write(name + "\n")

# ========== 开始转换 ==========
img_items = list(data['imgs'].items())
random.shuffle(img_items)
train_ratio = 0.8
split_index = int(len(img_items) * train_ratio)

for i, (img_id, img_info) in enumerate(tqdm(img_items)):
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

print("✅ 转换完成！YOLO格式数据保存在 yolo_dataset/")
