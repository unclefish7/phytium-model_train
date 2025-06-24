from torchinfo import summary
from models.experimental import attempt_load

model = attempt_load('C:/Users/zzy/Desktop/model_optimized.onnx', map_location='cpu')  # 加载模型
summary(model, input_size=(1, 3, 640, 640))  # 查看模型结构、参数量、通道数等
