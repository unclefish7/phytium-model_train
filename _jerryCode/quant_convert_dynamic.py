from onnxruntime.quantization import quantize_dynamic, QuantType

# 输入/输出路径
input_onnx = "model/yolov5n_960p.onnx"
output_onnx = "model/yolov5n_960p_int8_dynamic.onnx"

# 执行动态量化（仅权重）
quantize_dynamic(
    model_input=input_onnx,
    model_output=output_onnx,
    weight_type=QuantType.QInt8  # 权重量化为 int8，激活保持 float32
)

print(f"✅ 动态量化完成：{output_onnx}")
