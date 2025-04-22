import os
import json
import cv2
from matplotlib import pyplot as plt

# === 获取当前脚本所在目录 ===
base_dir = os.path.dirname(os.path.abspath(__file__))

# === 标注和图像路径 ===
annotation_path = os.path.join(base_dir, "../../data/annotations.json")  # 替换成你的 JSON 文件名
image_root = os.path.join(base_dir, "../../data")  # 图像主目录（包含 train/test/other）

# === 加载标注数据 ===
with open(annotation_path, "r") as f:
    data = json.load(f)

imgs = data.get("imgs", {})

# === 遍历图像和标注 ===
for img_id, img_info in imgs.items():
    image_path = os.path.join(image_root, img_info["path"])
    if not os.path.exists(image_path):
        print(f"图像不存在：{image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        continue

    # 绘制所有标注框
    for obj in img_info.get("objects", []):
        category = obj["category"]
        bbox = obj["bbox"]
        xmin, ymin = int(bbox["xmin"]), int(bbox["ymin"])
        xmax, ymax = int(bbox["xmax"]), int(bbox["ymax"])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, category, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 显示图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(f"{img_info['path']}")
    plt.axis("off")
    plt.show()

    input("按回车查看下一张图像，或 Ctrl+C 退出...")
