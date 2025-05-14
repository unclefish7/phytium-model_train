import os
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# ✅ 校准数据读取器
class Yolov5CalibReader(CalibrationDataReader):
    def __init__(self, calib_dir="data/calib"):
        self.image_paths = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.index = 0
        self.data = []

        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ 无法读取图像: {path}")
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                print(f"⚠️ 跳过非 RGB 图像: {path}")
                continue

            img = cv2.resize(img, (960, 960))               # 强制 resize 为 960x960
            img = img[:, :, ::-1]                           # BGR → RGB
            img = img.transpose(2, 0, 1).astype(np.float32) # HWC → CHW，转 float32
            img /= 255.0
            self.data.append({"images": np.expand_dims(img, axis=0)})  # shape = [1,3,960,960]

    def get_next(self):
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

# ✅ 执行静态量化
def quantize_onnx(model_fp32_path, model_int8_path, calib_dir="data/calib"):
    print("🧪 读取校准图像...")
    dr = Yolov5CalibReader(calib_dir)

    print("⚙️ 执行静态量化...")
    quantize_static(
        model_input=model_fp32_path,
        model_output=model_int8_path,
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    print(f"✅ 静态量化完成：{model_int8_path}")

if __name__ == "__main__":
    quantize_onnx(
        model_fp32_path="model/yolov5n_960p.onnx",
        model_int8_path="model/yolov5n_960p_int8.onnx",
        calib_dir="../data/calib"
    )