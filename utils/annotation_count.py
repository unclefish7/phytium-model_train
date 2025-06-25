import json
from collections import Counter

# ✅ 手动填写标注文件路径
annotation_path = r"E:\tt100k\phytium-model_train\utils\annotations.json"  # ← 改成你自己的路径

# 加载标注文件
with open(annotation_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计每个类别出现的次数
label_counter = Counter()

# 解析新的annotation格式：{"imgs": {图片id: {"objects": [标注列表], ...}, ...}}
for img_id, img_info in data["imgs"].items():
    objects = img_info.get("objects", [])
    for obj in objects:
        category = obj["category"]
        label_counter[category] += 1

# 获取前 N 个最常见类别
top_n = 20
sorted_labels = label_counter.most_common(top_n)

# 打印结果
print(f"前 {top_n} 个最常见的交通标志类别：")
for i, (label, count) in enumerate(sorted_labels, 1):
    print(f"{i:02d}. {label}: {count} 个标注")
