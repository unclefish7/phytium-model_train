# 在容器中的训练命令
```bash
python train.py --batch-size 16 --epochs 100 --name simple_net_finetune_ --cfg models/yolov5_mobilenetv4_small.yaml --data /workspace/dataset/tt100k.yaml --weights /workspace/model/latest_model_20250520/weights/best.pt --device 0 --image-weights --cos-lr --hyp data/hyps/hyp.scratch-low.yaml --img 1024
```

```bash
python train.py --batch-size 4 --epochs 200 --name simple_net_one_cls --cfg models/yolov5_mobilenetv4_small.yaml --data /workspace/dataset/tt100k.yaml --weights '' --device 0 --image-weights --hyp data/hyps/hyp.detect-only.yaml --img 2048
```

```bash
python detect.py --weights runs/train/simple_net_small_dataset_3/weights/best.pt --source /workspace/dataset/images/test  --img 2048
```

```bash
python val.py --weights runs/train/simple_net_small_dataset_3/weights/best.pt --data /workspace/dataset/tt100k.yaml --img 2048 --device 0 --batch-size 4
```