import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import Model
from models.mobilenetv4 import MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge
from utils.distillation import DistillationLoss
from utils.general import (LOGGER, check_dataset, check_file, check_img_size, check_yaml,
                           colorstr, increment_path)
from utils.torch_utils import select_device, time_sync
from train import train as original_train

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # 原始训练参数
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5n.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5_MobileNetv4_small.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    
    # 知识蒸馏参数
    parser.add_argument('--teacher-weights', type=str, default=ROOT / 'yolov5l.pt', help='teacher model weights path')
    parser.add_argument('--teacher-cfg', type=str, default='', help='teacher model.yaml path')
    parser.add_argument('--temperature', type=float, default=4.0, help='distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for feature distillation loss')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for classification distillation loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='weight for bbox regression distillation loss')
    parser.add_argument('--distill-weight', type=float, default=0.5, help='weight for distillation loss')
    
    # 其他参数
    parser.add_argument('--project', default=ROOT / 'runs/train-distill', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    opt = parser.parse_args()
    return opt

def train_with_distillation(opt):
    # 设置设备
    device = select_device(opt.device)
    
    # 加载配置
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
        
    # 检查数据集
    data_dict = check_dataset(opt.data)
    
    # 创建输出目录
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载教师模型
    LOGGER.info(f'Loading teacher model from {opt.teacher_weights}...')
    if opt.teacher_cfg:
        teacher_model = Model(opt.teacher_cfg)
    else:
        teacher_model = torch.load(opt.teacher_weights, map_location=device)['model'].float()
    teacher_model.to(device)
    teacher_model.eval()  # 设置为评估模式
    
    # 加载学生模型
    LOGGER.info(f'Loading student model from {opt.cfg}...')
    student_model = Model(opt.cfg)
    if opt.weights.endswith('.pt'):
        ckpt = torch.load(opt.weights, map_location=device)
        student_model.load_state_dict(ckpt['model'].state_dict(), strict=False)
    student_model.to(device)
    
    # 创建蒸馏损失
    distiller = DistillationLoss(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=opt.temperature,
        alpha=opt.alpha,
        beta=opt.beta,
        gamma=opt.gamma
    )
    
    # 创建TensorBoard记录器
    tb_writer = SummaryWriter(save_dir)
    
    # 修改Trainer类，加入知识蒸馏损失
    class DistillTrainer(original_train):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distiller = distiller
            self.distill_weight = opt.distill_weight
            
        def compute_loss(self, pred, targets, *args):
            # 原始监督损失
            orig_loss, loss_items = super().compute_loss(pred, targets, *args)
            
            # 教师模型的输出
            with torch.no_grad():
                teacher_outputs = self.distiller.teacher_model(pred[0])
                
            # 获取蒸馏损失
            distill_loss, distill_loss_items = self.distiller.get_distillation_loss(pred, teacher_outputs)
            
            # 总损失 = 原始损失 * (1 - distill_weight) + 蒸馏损失 * distill_weight
            total_loss = orig_loss * (1 - self.distill_weight) + distill_loss * self.distill_weight
            
            # 记录蒸馏损失到TensorBoard
            if self.tb_writer:
                for k, v in distill_loss_items.items():
                    self.tb_writer.add_scalar(f'Distillation/{k}', v, self.epoch)
                self.tb_writer.add_scalar('Distillation/total_loss', distill_loss.item(), self.epoch)
                    
            return total_loss, loss_items
    
    # 执行训练
    LOGGER.info(f"Starting knowledge distillation training for {opt.epochs} epochs...")
    trainer = DistillTrainer(
        opt=opt,
        device=device,
        hyp=hyp, 
        data_dict=data_dict,
        save_dir=save_dir,
        tb_writer=tb_writer
    )
    trainer.train()
    
if __name__ == "__main__":
    opt = parse_opt()
    train_with_distillation(opt)
