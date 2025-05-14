from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# 输入路径（你的模型）
input_model = "model/yolov5n_960p.onnx"
# 输出路径（量化后的模型）
output_model = "model/yolov5n_960p_int8.onnx"

# 检查输入文件是否存在
if not os.path.exists(input_model):
    print(f"❌ 未找到模型文件: {input_model}")
    exit(1)

# 执行动态量化（只量化权重，保持推理稳定性）
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # 权重量化为 int8
)

print(f"✅ 模型量化完成！已保存为: {output_model}")
