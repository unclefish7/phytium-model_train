# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
使用迁移学习后的小模型继续微调训练
使用方法：
    $ python train_small_model.py --weights yolov5_mobilenetv4_small.pt --cfg models/yolov5_MobileNetv4_small.yaml --data data/coco128.yaml --epochs 30 --batch-size 16
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到PATH

from train import train
from models.experimental import attempt_load
from utils.general import increment_path, get_latest_run, check_git_status, check_requirements, print_args
from utils.torch_utils import select_device

def parse_opt(known=False):
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='迁移学习后的模型权重')
    parser.add_argument('--cfg', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='数据集配置文件路径')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='超参数路径')
    parser.add_argument('--epochs', type=int, default=30, help='总训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='总批量大小')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, val]图片大小')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='从最后一个checkpoint恢复')
    parser.add_argument('--nosave', action='store_true', help='只保存最终checkpoint')
    parser.add_argument('--notest', action='store_true', help='只在最后一轮测试')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用autoanchor检查')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='进化超参数的次数')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='缓存图像以加快训练')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择进行训练')
    parser.add_argument('--device', default='', help='cuda设备, 如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--multi-scale', action='store_true', help='多尺度训练, 图片大小+/-50%%')
    parser.add_argument('--single-cls', action='store_true', help='将多类数据作为单类训练')
    parser.add_argument('--adam', action='store_true', help='使用adam优化器')
    parser.add_argument('--sync-bn', action='store_true', help='使用SyncBatchNorm, 只在DDP模式有效')
    parser.add_argument('--workers', type=int, default=8, help='dataloader最大worker数量')
    parser.add_argument('--project', default='runs/train', help='保存到project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='保存到project/name')
    parser.add_argument('--exist-ok', action='store_true', help='已存在的project/name，不要增加序号')
    parser.add_argument('--quad', action='store_true', help='对dataloader使用四元数据增强')
    parser.add_argument('--linear-lr', action='store_true', help='线性学习率')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='上传数据集到W&B')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='在W&B中设置边界框图像记录间隔')
    parser.add_argument('--save_period', type=int, default=-1, help='每x个epoch记录模型')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B构件的版本')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的DDP参数')
    parser.add_argument('--freeze', type=int, default=0, help='冻结层的数量')
    parser.add_argument('--patience', type=int, default=30, help='早停的EarlyStopping patience (epochs without improvement)')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    
    return opt

def main(opt):
    """主函数，用于启动训练"""
    # 检查设备和requirements
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # 运行训练
    train(opt.hyp, opt, device)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
