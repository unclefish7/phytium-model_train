import os
import cv2
import numpy as np
import time

# === 路径配置 ===
# MODEL_PATH = "model/yolov5n_960p_simplify.onnx"
MODEL_PATH = "model/yolov5n_960p.onnx"
INPUT_DIR = "resources/camera_sim"
OUTPUT_DIR = "resources/result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载模型 ===
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# === 后处理函数 ===
def postprocess(outputs, image, conf_thresh=0.4):
    boxes, class_ids, confidences = [], [], []

    for i in range(outputs.shape[1]):
        row = outputs[0][i]
        confidence = row[4]
        if confidence > conf_thresh:
            scores = row[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > conf_thresh:
                cx, cy, w, h = row[:4]
                left = int((cx - w/2) * image.shape[1] / 960)
                top = int((cy - h/2) * image.shape[0] / 960)
                width = int(w * image.shape[1] / 960)
                height = int(h * image.shape[0] / 960)

                boxes.append([left, top, width, height])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.5)
    if len(indices) == 0:
        return image

    for i in indices:
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[idx]
        label = f"ID:{class_ids[idx]} {confidences[idx]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
        cv2.putText(image, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return image

# === 获取图像列表 ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# === 统计总时长变量 ===
total_read_time = 0
total_pre_time = 0
total_infer_time = 0
total_post_time = 0
total_count = len(image_files)

print(f"📂 共准备推理 {total_count} 张图像...")

# === 主处理循环 ===
for file_name in image_files:
    t0 = time.time()
    image_path = os.path.join(INPUT_DIR, file_name)
    image = cv2.imread(image_path)
    t1 = time.time()
    resized = cv2.resize(image, (960, 960))
    blob = cv2.dnn.blobFromImage(resized, 1/255.0, (960, 960), swapRB=True, crop=False)
    t2 = time.time()
    net.setInput(blob)
    outputs = net.forward()
    t3 = time.time()
    result = postprocess(outputs, image)
    output_path = os.path.join(OUTPUT_DIR, f"det_{file_name}")
    cv2.imwrite(output_path, result)
    t4 = time.time()

    # 分阶段耗时统计
    read_time = t1 - t0
    pre_time = t2 - t1
    infer_time = t3 - t2
    post_time = t4 - t3

    total_read_time += read_time
    total_pre_time += pre_time
    total_infer_time += infer_time
    total_post_time += post_time

    print(f"✅ {file_name} | 读取: {read_time:.3f}s | 预处理: {pre_time:.3f}s | 推理: {infer_time:.3f}s | 后处理: {post_time:.3f}s")

# === 平均耗时统计 ===
print("\n📊 处理完成！平均每张图像耗时：")
print(f"📥 图像读取       : {total_read_time / total_count:.3f} 秒")
print(f"🔧 图像预处理     : {total_pre_time / total_count:.3f} 秒")
print(f"🧠 模型推理       : {total_infer_time / total_count:.3f} 秒")
print(f"🖼️ 结果后处理保存 : {total_post_time / total_count:.3f} 秒")
print(f"⏱️ 平均总耗时     : {(total_read_time + total_pre_time + total_infer_time + total_post_time) / total_count:.3f} 秒")
