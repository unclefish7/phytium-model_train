"""
TT100K 交通标志检测+分类数据集转换工具

功能说明:
- 将 TT100K 数据集转换为 YOLO 格式，用于目标检测任务
- 与纯检测版本不同，此版本保留具体的类别信息而非统一标记为单一类别
- 对样本量过少的类别进行智能合并，提高训练效果
- 生成的数据集既可以检测交通标志位置，也可以识别具体类别

输入文件:
- data/annotations.json: TT100K 标注文件
- data/: 图像目录

输出文件:
- yolo_dataset_detect_classify/: YOLO 格式数据集
  - images/train/: 训练图像
  - images/val/: 验证图像  
  - labels/train/: 训练标注
  - labels/val/: 验证标注
  - traffic_sign_detect_classify.yaml: 数据集配置文件
  - traffic_sign.names: 类别名称文件

类别处理策略:
1. 样本量 >= MERGE_THRESHOLD 的类别保持独立
2. 样本量 < MERGE_THRESHOLD 的同类型类别合并 (如 p50, p60 合并为 po)
3. 合并后仍 < MIN_SAMPLES_PER_CLASS 的类别进一步合并为 'o' 类别
4. 最终过滤掉样本量仍不足的类别

使用方法:
python tt100k_2_yolo_detect_and_classify.py
"""

import os
import json
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# ========== 配置参数 ==========
# 项目根目录配置 - 可根据实际情况修改
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# 输入路径配置
IMAGE_ROOT = os.path.join(PROJECT_ROOT, 'data')
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, 'data/annotations.json')

# 输出路径配置
OUTPUT_DATASET_NAME = 'yolo_dataset_detect_classify'
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, OUTPUT_DATASET_NAME)
YOLO_IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'images')
YOLO_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels')

# 类别处理参数配置（照搬classify脚本参数）
MIN_SAMPLES_PER_CLASS = 200  # 每个类别的最小样本数量，低于此数量的类别将被过滤
MERGE_THRESHOLD = 200        # 类别合并阈值，低于此数量的类别会尝试合并

# 数据集划分参数
TRAIN_RATIO = 0.8

# 创建输出目录
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_IMAGE_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(YOLO_LABEL_DIR, 'val'), exist_ok=True)

print("=" * 80)
print("🚦 TT100K Traffic Sign Detection+Classification Dataset Converter")
print("=" * 80)
print(f"🏠 Project root: {PROJECT_ROOT}")
print(f"📂 Input image directory: {IMAGE_ROOT}")
print(f"📄 Annotation file: {ANNOTATION_FILE}")
print(f"📁 Output dataset directory: {OUTPUT_ROOT}")
print("🎯 Task type: Traffic sign detection + classification")
print(f"🔧 Category processing: merge_threshold={MERGE_THRESHOLD}, min_samples={MIN_SAMPLES_PER_CLASS}")
print("-" * 80)

# 检查输入文件是否存在
if not os.path.exists(ANNOTATION_FILE):
    print(f"❌ Error: Annotation file not found {ANNOTATION_FILE}")
    exit(1)

if not os.path.exists(IMAGE_ROOT):
    print(f"❌ Error: Image directory not found {IMAGE_ROOT}")
    exit(1)

# ========== 加载标注数据 ==========
print("📖 Loading annotation file...")
try:
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Successfully loaded annotation file with {len(data['imgs'])} images")
except Exception as e:
    print(f"❌ Failed to load annotation file: {e}")
    exit(1)

# ========== 统计类别信息 ==========
print("📊 Analyzing category statistics...")
category_stats = defaultdict(int)
total_objects = 0

for img_id, img_info in data['imgs'].items():
    if img_info.get('objects'):
        for obj in img_info['objects']:
            category = obj.get('category', 'unknown')
            category_stats[category] += 1
            total_objects += 1

print(f"📈 Found {len(category_stats)} categories with {total_objects} total annotations")
print("🏷️  Top 10 categories by frequency:")
sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
for i, (category, count) in enumerate(sorted_categories[:10]):
    print(f"   {i+1:2d}. {category:15s}: {count:6d} samples")
if len(sorted_categories) > 10:
    print(f"   ... and {len(sorted_categories) - 10} more categories")

# ========== 类别合并处理 ==========
print(f"\n🔄 Processing category merging (threshold: {MERGE_THRESHOLD})...")

def extract_base_category(category):
    """提取类别的基础名称（去除数字）"""
    import re
    # 匹配字母开头，后面跟数字的模式
    match = re.match(r'^([a-zA-Z]+)', category)
    if match:
        return match.group(1)
    return category

# 创建类别映射表（原类别 -> 合并后类别）
category_mapping = {}
merged_category_stats = defaultdict(int)

# 先识别需要合并的类别
categories_to_merge = {}  # 基础类别名 -> [具体类别列表]
standalone_categories = {}  # 独立类别

for category, count in category_stats.items():
    base_category = extract_base_category(category)
    
    if count >= MERGE_THRESHOLD:
        # 样本量足够，作为独立类别
        standalone_categories[category] = count
        category_mapping[category] = category
        merged_category_stats[category] = count
    else:
        # 样本量不足，需要合并
        if base_category not in categories_to_merge:
            categories_to_merge[base_category] = []
        categories_to_merge[base_category].append((category, count))

# 处理需要合并的类别
merge_info = []
for base_category, category_list in categories_to_merge.items():
    # 计算合并后的总数量
    total_count = sum(count for _, count in category_list)
    
    # 确定合并后的类别名称
    merged_name = base_category + 'o'
    
    # 检查是否与已有的独立类别冲突
    if merged_name in standalone_categories:
        # 如果冲突，合并到已有类别
        merged_category_stats[merged_name] += total_count
        standalone_categories[merged_name] += total_count
    else:
        # 创建新的合并类别
        merged_category_stats[merged_name] = total_count
    
    # 更新映射关系
    for category, count in category_list:
        category_mapping[category] = merged_name
    
    merge_info.append((base_category, category_list, merged_name, total_count))

# 显示合并信息
if merge_info:
    print("🔗 Category merging information:")
    for base_category, category_list, merged_name, total_count in merge_info:
        category_names = [f"{cat}({cnt})" for cat, cnt in category_list]
        print(f"   {base_category}: {', '.join(category_names)} -> {merged_name}({total_count})")
else:
    print("📋 No categories need merging")

print("📊 Post-merge statistics:")
print(f"   - Pre-merge categories: {len(category_stats)}")
print(f"   - Post-merge categories: {len(merged_category_stats)}")
print(f"   - Standalone categories: {len(standalone_categories)}")
print(f"   - Merge groups: {len(merge_info)}")

# ========== 过滤样本量过少的类别 ==========
print(f"\n🔍 Filtering categories with < {MIN_SAMPLES_PER_CLASS} samples...")
valid_categories = {cat: count for cat, count in merged_category_stats.items() if count >= MIN_SAMPLES_PER_CLASS}
insufficient_categories = {cat: count for cat, count in merged_category_stats.items() if count < MIN_SAMPLES_PER_CLASS}

print(f"✅ Retained categories: {len(valid_categories)}")
print(f"❌ Insufficient categories: {len(insufficient_categories)}")

# ========== 二次合并处理 ==========
if insufficient_categories:
    print("\n🔄 Performing secondary merge (merging insufficient categories to 'o' category)...")
    
    # 计算合并到 'o' 类别的总数量
    total_o_count = sum(insufficient_categories.values())
    
    print("🔗 Secondary merge information:")
    print(f"   Merging categories: {list(insufficient_categories.keys())}")
    print(f"   Total count before merge: {total_o_count}")
    
    # 检查是否已存在 'o' 类别
    if 'o' in valid_categories:
        # 如果已存在，合并到现有的 'o' 类别
        valid_categories['o'] += total_o_count
        print(f"   Merged to existing 'o' category, new count: {valid_categories['o']}")
    else:
        # 如果不存在，创建新的 'o' 类别
        valid_categories['o'] = total_o_count
        print(f"   Created new 'o' category with count: {valid_categories['o']}")
    
    # 更新类别映射：将所有数量不足的类别映射到 'o'
    for insufficient_cat in insufficient_categories.keys():
        # 找到所有映射到这个不足类别的原始类别，重新映射到 'o'
        for orig_cat, mapped_cat in category_mapping.items():
            if mapped_cat == insufficient_cat:
                category_mapping[orig_cat] = 'o'
    
    # 清空不足类别列表（因为已经合并了）
    filtered_categories = {}
    
    print("📊 Post-secondary merge statistics:")
    print(f"   - Final valid categories: {len(valid_categories)}")
    print("   - Completely filtered categories: 0")
    print(f"   - 'o' category sample count: {valid_categories.get('o', 0)}")
else:
    filtered_categories = insufficient_categories
    print("\n📋 No secondary merge needed")

if filtered_categories:
    print("🗑️  Filtered categories:")
    sorted_filtered = sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_filtered[:10]:  # 显示前10个被过滤的类别
        print(f"   {category:15s}: {count:6d} samples (< {MIN_SAMPLES_PER_CLASS})")
    if len(sorted_filtered) > 10:
        print(f"   ... and {len(sorted_filtered) - 10} more filtered categories")

# ========== 创建类别索引映射 ==========
print("\n📋 Creating category index mapping...")
sorted_valid_categories = sorted(valid_categories.keys())
category_to_index = {cat: idx for idx, cat in enumerate(sorted_valid_categories)}

print("📊 Final category summary:")
total_valid_objects = sum(valid_categories.values())
print(f"   - Total valid categories: {len(valid_categories)}")
print(f"   - Total valid annotations: {total_valid_objects}")
print(f"   - Filtered annotations: {total_objects - total_valid_objects}")

print("🏷️  Final category list with indices:")
for idx, category in enumerate(sorted_valid_categories):
    print(f"   {idx:2d}: {category:15s} ({valid_categories[category]:6d} samples)")

# ========== 创建配置文件 ==========
print("\n📄 Creating configuration files...")

# 创建类别名称文件
names_file_path = os.path.join(OUTPUT_ROOT, 'traffic_sign.names')
with open(names_file_path, "w", encoding='utf-8') as f:
    for category in sorted_valid_categories:
        f.write(f"{category}\n")

# 创建 YOLO 配置文件
yaml_content = f"""# YOLO dataset configuration for traffic sign detection + classification
# Dataset path
path: {OUTPUT_ROOT}

# Train and validation sets
train: images/train
val: images/val

# Number of classes
nc: {len(valid_categories)}

# Class names
names: {sorted_valid_categories}
"""

yaml_file_path = os.path.join(OUTPUT_ROOT, 'traffic_sign_detect_classify.yaml')
with open(yaml_file_path, "w", encoding='utf-8') as f:
    f.write(yaml_content)

print(f"📋 Category names file created: {names_file_path}")
print(f"⚙️  Configuration file created: {yaml_file_path}")

# ========== 开始转换数据 ==========
print("\n🔄 Starting data conversion...")

# 筛选包含有效标注的图片
valid_img_items = []
total_valid_annotations = 0

for img_id, img_info in data['imgs'].items():
    if img_info.get('objects'):
        # 检查该图片是否包含有效类别的标注
        valid_objects = []
        for obj in img_info['objects']:
            original_category = obj.get('category', 'unknown')
            if original_category in category_mapping:
                mapped_category = category_mapping[original_category]
                if mapped_category in valid_categories:
                    valid_objects.append(obj)
        
        if valid_objects:
            # 更新img_info中的objects为有效对象
            img_info_copy = img_info.copy()
            img_info_copy['objects'] = valid_objects
            valid_img_items.append((img_id, img_info_copy))
            total_valid_annotations += len(valid_objects)

print("📊 Dataset statistics:")
print(f"   - Images with valid annotations: {len(valid_img_items)}")
print(f"   - Total valid annotations: {total_valid_annotations}")
print(f"   - Average annotations per image: {total_valid_annotations/len(valid_img_items):.2f}")

# 随机打乱并划分训练/验证集
random.seed(42)  # 设置随机种子以确保结果可复现
random.shuffle(valid_img_items)
split_index = int(len(valid_img_items) * TRAIN_RATIO)

print("\n📂 Dataset split:")
print(f"   - Training set: {split_index} images")
print(f"   - Validation set: {len(valid_img_items) - split_index} images")
print(f"   - Train/val ratio: {TRAIN_RATIO:.1f}/{1-TRAIN_RATIO:.1f}")

# 开始处理图片
print("\n🚀 Processing images...")
processed_count = 0
error_count = 0
category_annotation_counts = defaultdict(int)

for i, (img_id, img_info) in enumerate(tqdm(valid_img_items, desc="Processing images")):
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
            print(f"⚠️  Cannot read image dimensions: {src_img_path}, error: {e}")
            error_count += 1
            continue

        # 生成 YOLO 标注文件
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(YOLO_LABEL_DIR, phase, label_filename)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in img_info['objects']:
                original_category = obj.get('category', 'unknown')
                
                # 获取映射后的类别
                if original_category in category_mapping:
                    mapped_category = category_mapping[original_category]
                else:
                    mapped_category = original_category
                
                # 跳过无效类别
                if mapped_category not in valid_categories:
                    continue
                
                # 获取类别索引
                cls_id = category_to_index[mapped_category]
                category_annotation_counts[mapped_category] += 1
                
                bbox = obj['bbox']
                xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                
                # 转换为 YOLO 格式（归一化坐标）
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height
                
                # 写入标注（格式：class_id x_center y_center width height）
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        
        processed_count += 1
        
        # 每处理1000张图片显示一次进度
        if (processed_count + 1) % 1000 == 0:
            print(f"📈 Processed {processed_count + 1} images...")
            
    except Exception as e:
        print(f"⚠️  Error processing image: {img_info.get('path', 'unknown')}, error: {e}")
        error_count += 1
        continue

# ========== 转换完成 ==========
print("\n" + "=" * 80)
print("✅ TT100K Traffic Sign Detection+Classification Dataset Conversion Complete!")
print("=" * 80)
print(f"📁 Output directory: {OUTPUT_ROOT}")
print(f"📋 Category names file: {names_file_path}")
print(f"⚙️  Configuration file: {yaml_file_path}")
print("📊 Processing results:")
print(f"   - Successfully processed images: {processed_count}")
print(f"   - Error/skipped images: {error_count}")
print(f"   - Training set images: {min(split_index, processed_count)}")
print(f"   - Validation set images: {max(0, processed_count - split_index)}")
print(f"   - Total valid categories: {len(valid_categories)}")

print("\n🏷️  Annotation count per category:")
print("-" * 50)
sorted_annotation_counts = sorted(category_annotation_counts.items(), key=lambda x: x[1], reverse=True)
total_annotations_processed = 0
for category, count in sorted_annotation_counts:
    print(f"   {category:20s}: {count:6d} annotations")
    total_annotations_processed += count

print("-" * 50)
print(f"   {'Total':20s}: {total_annotations_processed:6d} annotations")

print("\n🎯 Dataset usage:")
print("   - Task: Traffic sign detection + classification")
print("   - Format: YOLO (class_id x_center y_center width height)")
print(f"   - Categories merged with threshold: {MERGE_THRESHOLD}")
print(f"   - Minimum samples per category: {MIN_SAMPLES_PER_CLASS}")
print("   - Low-sample categories merged into base category + 'o' suffix")
print("   - Very low-sample categories further merged into 'o' category")
print("\n💡 Note: This dataset can both detect traffic sign positions and classify specific categories")
print("=" * 80)
