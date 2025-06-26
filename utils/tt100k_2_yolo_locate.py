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
OUTPUT_DATASET_NAME = 'yolo_dataset_one_cls'
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, OUTPUT_DATASET_NAME)
YOLO_IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'images')
YOLO_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels')

# 创建输出目录
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'val'), exist_ok=True)

print("=" * 60)
print("🚦 TT100K 交通标志位置检测数据集转换工具")
print("=" * 60)
print(f"🏠 项目根目录: {PROJECT_ROOT}")
print(f"📂 输入数据目录: {IMAGE_ROOT}")
print(f"📄 标注文件: {ANNOTATION_FILE}")
print(f"📁 输出数据集目录: {OUTPUT_ROOT}")
print("🎯 任务类型: 交通标志位置检测 (所有类别统一为 class_id=0)")
print("-" * 60)

# 检查输入文件是否存在
if not os.path.exists(ANNOTATION_FILE):
    print(f"❌ 错误: 标注文件不存在 {ANNOTATION_FILE}")
    exit(1)

if not os.path.exists(IMAGE_ROOT):
    print(f"❌ 错误: 图像目录不存在 {IMAGE_ROOT}")
    exit(1)

# ========== 加载标注 ==========
print("📖 正在加载标注文件...")
try:
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 成功加载标注文件，包含 {len(data['imgs'])} 张图像")
except Exception as e:
    print(f"❌ 加载标注文件失败: {e}")
    exit(1)

# 创建类别映射文件（只有一个类别：交通标志）
names_file_path = os.path.join(OUTPUT_ROOT, 'traffic_sign.names')
with open(names_file_path, "w", encoding='utf-8') as f:
    f.write("traffic_sign\n")

# 创建 YOLO 配置文件
yaml_content = f"""# YOLO dataset configuration for traffic sign detection
# Dataset path
path: {OUTPUT_ROOT}

# Train and validation sets
train: images/train
val: images/val

# Number of classes (only position detection, no classification)
nc: 1

# Class names
names: ['traffic_sign']
"""

yaml_file_path = os.path.join(OUTPUT_ROOT, 'traffic_sign_detection.yaml')
with open(yaml_file_path, "w", encoding='utf-8') as f:
    f.write(yaml_content)

print(f"📋 类别文件已创建: {names_file_path}")
print(f"⚙️  配置文件已创建: {yaml_file_path}")

# ========== 开始转换 ==========
print("\n🔄 开始转换数据...")

# 筛选包含标注的图片（任何类别的交通标志都保留）
valid_img_items = []
total_annotations = 0

for img_id, img_info in data['imgs'].items():
    # 检查该图片是否包含任何标注
    if img_info.get('objects') and len(img_info['objects']) > 0:
        valid_img_items.append((img_id, img_info))
        total_annotations += len(img_info['objects'])

print("📊 统计信息:")
print(f"   - 包含标注的图片数量: {len(valid_img_items)}")
print(f"   - 标注总数: {total_annotations}")
print(f"   - 平均每张图片标注数: {total_annotations/len(valid_img_items):.2f}")

# 随机打乱并划分训练/验证集
random.seed(42)  # 设置随机种子以确保结果可复现
random.shuffle(valid_img_items)
train_ratio = 0.8
split_index = int(len(valid_img_items) * train_ratio)

print("\n📂 数据集划分:")
print(f"   - 训练集: {split_index} 张图片")
print(f"   - 验证集: {len(valid_img_items) - split_index} 张图片")
print(f"   - 训练/验证比例: {train_ratio:.1f}/{1-train_ratio:.1f}")

# 开始处理图片
print("\n🚀 开始处理图片...")
processed_count = 0
error_count = 0

for i, (img_id, img_info) in enumerate(tqdm(valid_img_items, desc="处理图片")):
    try:
        phase = 'train' if i < split_index else 'val'
        src_img_path = os.path.join(IMAGE_ROOT, img_info['path'])
        
        # 检查源图片是否存在
        if not os.path.exists(src_img_path):
            error_count += 1
            continue

        # 复制图像
        img_filename = os.path.basename(src_img_path)
        dst_img_path = os.path.join(YOLO_IMAGE_DIR, phase, img_filename)
        shutil.copyfile(src_img_path, dst_img_path)

        # 获取图像尺寸
        try:
            with Image.open(src_img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️  无法读取图像尺寸: {src_img_path}, 错误: {e}")
            error_count += 1
            continue

        # 生成 YOLO 标注文件
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(YOLO_LABEL_DIR, phase, label_filename)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in img_info['objects']:
                # 统一类别 ID 为 0（交通标志位置检测）
                cls = 0
                bbox = obj['bbox']
                xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                
                # 转换为 YOLO 格式（归一化坐标）
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height
                
                # 写入标注（格式：class_id x_center y_center width height）
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        
        processed_count += 1
        
        # 每处理1000张图片显示一次进度
        if (processed_count + 1) % 1000 == 0:
            print(f"📈 已处理 {processed_count + 1} 张图片...")
            
    except Exception as e:
        print(f"⚠️  处理图片时出错: {img_info.get('path', 'unknown')}, 错误: {e}")
        error_count += 1
        continue

# ========== 转换完成 ==========
print("\n" + "=" * 60)
print("✅ TT100K 交通标志位置检测数据集转换完成！")
print("=" * 60)
print(f"📁 输出目录: {OUTPUT_ROOT}")
print(f"📋 类别文件: {names_file_path}")
print(f"⚙️  配置文件: {yaml_file_path}")
print("📊 处理结果:")
print(f"   - 成功处理图片: {processed_count} 张")
print(f"   - 错误/跳过图片: {error_count} 张")
print(f"   - 训练集图片: {min(split_index, processed_count)} 张")
print(f"   - 验证集图片: {max(0, processed_count - split_index)} 张")
print("🎯 数据集用途: 交通标志位置检测（不分类别）")
print("🏷️  所有标注统一标记为: class_id = 0")
print("\n💡 提示: 使用此数据集训练的模型只能检测交通标志的位置，不能识别具体类别")
print("=" * 60)