import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ========== 配置参数 ==========
# 项目根目录配置 - 可根据实际情况修改
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# 输入路径配置
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data')
ANNO_PATH = os.path.join(PROJECT_ROOT, 'data/annotations.json')

# 输出路径配置
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'yolo_dataset_classify')

# 裁剪参数配置
PADDING_RATIO = 0.1  # 默认加10%的padding
MIN_WIDTH = 16       # 最小宽度（像素）
MIN_HEIGHT = 16      # 最小高度（像素）

# 输出图像尺寸配置
OUTPUT_WIDTH = 128   # 输出图像宽度（像素）
OUTPUT_HEIGHT = 128  # 输出图像高度（像素）

# 类别过滤配置
MIN_SAMPLES_PER_CLASS = 50  # 每个类别的最小样本数量，低于此数量的类别将被过滤

print("=" * 80)
print("🚦 TT100K 交通标志分类数据集生成工具")
print("=" * 80)
print(f"🏠 项目根目录: {PROJECT_ROOT}")
print(f"📂 图像目录: {IMAGE_DIR}")
print(f"📄 标注文件: {ANNO_PATH}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"🔧 裁剪参数: padding={PADDING_RATIO*100:.0f}%, 最小尺寸={MIN_WIDTH}x{MIN_HEIGHT}")
print(f"📐 输出图像尺寸: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
print(f"🔍 类别过滤: 最少样本数={MIN_SAMPLES_PER_CLASS}")
print("-" * 80)

# 检查输入文件是否存在
if not os.path.exists(ANNO_PATH):
    print(f"❌ 错误: 标注文件不存在 {ANNO_PATH}")
    exit(1)

if not os.path.exists(IMAGE_DIR):
    print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")
    exit(1)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 加载标注数据 ==========
print("📖 正在加载标注文件...")
try:
    with open(ANNO_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 成功加载标注文件，包含 {len(data['imgs'])} 张图像")
except Exception as e:
    print(f"❌ 加载标注文件失败: {e}")
    exit(1)

# ========== 统计类别信息 ==========
print("📊 正在统计类别信息...")
category_stats = defaultdict(int)
total_objects = 0

for img_id, img_info in data['imgs'].items():
    if img_info.get('objects'):
        for obj in img_info['objects']:
            category = obj.get('category', 'unknown')
            category_stats[category] += 1
            total_objects += 1

print(f"📈 找到 {len(category_stats)} 个类别，共 {total_objects} 个标注对象")
print("🏷️  类别统计（前10个最多的类别）:")
sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
for i, (category, count) in enumerate(sorted_categories[:10]):
    print(f"   {i+1:2d}. {category:15s}: {count:6d} 个")
if len(sorted_categories) > 10:
    print(f"   ... 还有 {len(sorted_categories) - 10} 个类别")

# ========== 过滤样本量过少的类别 ==========
print(f"\n🔍 正在过滤样本量少于 {MIN_SAMPLES_PER_CLASS} 的类别...")
valid_categories = {cat: count for cat, count in category_stats.items() if count >= MIN_SAMPLES_PER_CLASS}
filtered_categories = {cat: count for cat, count in category_stats.items() if count < MIN_SAMPLES_PER_CLASS}

print(f"✅ 保留类别: {len(valid_categories)} 个")
print(f"❌ 过滤类别: {len(filtered_categories)} 个")

if filtered_categories:
    print("🗑️  被过滤的类别:")
    sorted_filtered = sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_filtered[:10]:  # 显示前10个被过滤的类别
        print(f"   {category:15s}: {count:6d} 个 (< {MIN_SAMPLES_PER_CLASS})")
    if len(sorted_filtered) > 10:
        print(f"   ... 还有 {len(sorted_filtered) - 10} 个被过滤的类别")

print("📊 有效类别统计:")
total_valid_objects = sum(valid_categories.values())
print(f"   - 有效类别总数: {len(valid_categories)}")
print(f"   - 有效标注总数: {total_valid_objects}")
print(f"   - 过滤掉的标注: {total_objects - total_valid_objects}")

# ========== 创建类别目录 ==========
print("\n📁 正在创建类别目录...")
for category in valid_categories.keys():
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
print(f"✅ 创建了 {len(valid_categories)} 个有效类别目录")

# ========== 裁剪并保存图像 ==========
print("\n🔄 开始裁剪和保存图像...")

# 统计变量
processed_images = 0
processed_objects = 0
skipped_objects = 0
error_images = 0
category_counts = defaultdict(int)

def crop_with_padding(image, bbox, padding_ratio=0.1):
    """
    从图像中裁剪指定区域，并添加padding，然后resize到指定尺寸
    
    Args:
        image: 输入图像 (H, W, C)
        bbox: 边界框 [xmin, ymin, xmax, ymax]
        padding_ratio: padding比例
    
    Returns:
        裁剪并resize后的图像，如果失败返回None
    """
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    
    # 计算原始框的宽高
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # 添加padding
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # 扩展边界框
    new_xmin = max(0, xmin - pad_w)
    new_ymin = max(0, ymin - pad_h)
    new_xmax = min(w, xmax + pad_w)
    new_ymax = min(h, ymax + pad_h)
    
    # 检查裁剪区域是否有效
    crop_w = new_xmax - new_xmin
    crop_h = new_ymax - new_ymin
    
    if crop_w < MIN_WIDTH or crop_h < MIN_HEIGHT:
        return None
    
    # 裁剪图像
    cropped = image[new_ymin:new_ymax, new_xmin:new_xmax]
    
    # 调整图像大小到指定尺寸
    resized = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    return resized

# 处理所有图像
for img_id, img_info in tqdm(data['imgs'].items(), desc="处理图像"):
    # 跳过没有标注的图像
    if not img_info.get('objects'):
        continue
    
    # 构建图像路径
    img_path = os.path.join(IMAGE_DIR, img_info['path'])
    
    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        error_images += 1
        continue
    
    try:
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  无法读取图像: {img_path}")
            error_images += 1
            continue
        
        processed_images += 1
        
        # 获取图像文件名（不含扩展名）
        img_basename = os.path.splitext(os.path.basename(img_info['path']))[0]
        
        # 处理该图像中的所有标注对象
        for obj_idx, obj in enumerate(img_info['objects']):
            category = obj.get('category', 'unknown')
            
            # 跳过样本量过少的类别
            if category not in valid_categories:
                skipped_objects += 1
                continue
                
            bbox_info = obj.get('bbox', {})
            
            # 检查bbox信息是否完整
            required_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            if not all(key in bbox_info for key in required_keys):
                skipped_objects += 1
                continue
            
            # 提取边界框坐标
            xmin = int(bbox_info['xmin'])
            ymin = int(bbox_info['ymin'])
            xmax = int(bbox_info['xmax'])
            ymax = int(bbox_info['ymax'])
            
            # 检查边界框是否有效
            if xmin >= xmax or ymin >= ymax:
                skipped_objects += 1
                continue
            
            # 检查边界框是否在图像范围内
            h, w = image.shape[:2]
            if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                # 将边界框限制在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
            
            # 再次检查修正后的边界框
            if xmin >= xmax or ymin >= ymax:
                skipped_objects += 1
                continue
            
            # 裁剪图像
            cropped_image = crop_with_padding(image, [xmin, ymin, xmax, ymax], PADDING_RATIO)
            
            if cropped_image is None:
                skipped_objects += 1
                continue
            
            # 构建保存路径
            category_dir = os.path.join(OUTPUT_DIR, category)
            filename = f"{img_basename}_{obj_idx}.jpg"
            save_path = os.path.join(category_dir, filename)
            
            # 保存裁剪后的图像
            success = cv2.imwrite(save_path, cropped_image)
            
            if success:
                processed_objects += 1
                category_counts[category] += 1
            else:
                print(f"⚠️  保存图像失败: {save_path}")
                skipped_objects += 1
        
    except Exception as e:
        print(f"⚠️  处理图像时出错 {img_path}: {e}")
        error_images += 1
        continue

# ========== 输出统计结果 ==========
print("\n" + "=" * 80)
print("✅ TT100K 交通标志分类数据集生成完成！")
print("=" * 80)
print(f"📁 输出目录: {OUTPUT_DIR}")
print("📊 处理统计:")
print(f"   - 处理图像总数: {processed_images}")
print(f"   - 错误/跳过图像: {error_images}")
print(f"   - 成功裁剪对象: {processed_objects}")
print(f"   - 跳过无效对象: {skipped_objects}")
print(f"   - 有效类别数: {len(category_counts)}")
print(f"   - 过滤类别数: {len(filtered_categories)}")

print("\n🏷️  各类别图像数量统计:")
print("-" * 50)
sorted_category_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
total_saved = 0
for category, count in sorted_category_counts:
    print(f"   {category:20s}: {count:6d} 张")
    total_saved += count

print("-" * 50)
print(f"   {'总计':20s}: {total_saved:6d} 张")

print("\n💡 数据集使用说明:")
print("   - 每个类别的图像保存在对应的子目录中")
print("   - 图像命名格式: <原图名>_<对象编号>.jpg")
print(f"   - 裁剪时添加了 {PADDING_RATIO*100:.0f}% 的边距")
print(f"   - 过滤了尺寸小于 {MIN_WIDTH}x{MIN_HEIGHT} 的对象")
print(f"   - 过滤了样本数少于 {MIN_SAMPLES_PER_CLASS} 的类别")
print(f"   - 所有输出图像统一调整为 {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} 尺寸")
print("=" * 80)