# 训练命令
```bash
python train.py --batch-size 16 --epochs 100 --cfg models/yolov5_mobilenetv4_small.yaml --data ../../yolo_dataset/tt100k.yaml --weights ../model/latest_model_20250520/weights/best.pt --device 0 --image-weights --cos-lr --hyp data/hyps/hyp.scratch-low.yaml
```