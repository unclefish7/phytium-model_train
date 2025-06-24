# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
将训练好的MobileNetV4模型迁移到小模型上（width_multiple=0.25）
使用方法：
    $ python transfer_to_small_model.py --weights yolov5_mobilenetv4.pt --cfg models/yolov5_MobileNetv4_small.yaml --save-path yolov5_mobilenetv4_small.pt
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到PATH

from models.common import *
from models.experimental import *
from models.mobilenetv4 import *
from models.yolo import Model
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_yaml, make_divisible, print_args
from utils.torch_utils import select_device

LOGGER.setLevel(logging.INFO)  # 设置日志级别为INFO

def parse_model(d, ch):  # model_dict, input_channels(3)
    """解析模型配置字典"""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # 每个anchor的输出数量

    layers, save, c2 = [], [], ch[-1]  # 层, 保存, 输出通道
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        try:
            m = eval(m) if isinstance(m, str) else m  # eval strings
        except:
            pass
        
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 根据深度倍数进行缩放
        if m in [Conv, Bottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, ConvTranspose, CrossConv, C3, C3TR, C2f]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是最终输出层
                c2 = make_divisible(c2 * gw, 8)  # 根据宽度倍数进行缩放

            args = [c1, c2, *args[1:]]
            if m in [C3, C3TR, C2f]:
                args.insert(2, n)  # 添加重复次数参数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is Detect:
            args.append([ch[x] for x in f])  # channels list
        elif m is MobileNetV4ConvSmall:
            c2 = [None, 32, 64, 96, 128]  # 该网络的四个输出通道
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 重复n次模块
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 添加索引, from, 类型, 参数数量等属性
        LOGGER.info(f'{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # 记录日志信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加需要保存的层
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def load_pretrained_model(weights, device):
    """加载预训练模型"""
    ckpt = torch.load(weights, map_location=device)  # 加载权重文件
    model_state_dict = ckpt['model'].state_dict()
    return model_state_dict, ckpt

def create_small_model(cfg_path, ch=3, device=''):
    """创建小模型"""
    cfg = check_yaml(cfg_path)  # 检查YAML文件
    with open(cfg, encoding='ascii', errors='ignore') as f:
        cfg_dict = yaml.safe_load(f)  # 加载模型配置
    
    model = Model(cfg_dict, ch=ch).to(device)  # 创建小模型
    return model

def transfer_weights(source_weights, target_model, device):
    """将权重从源模型迁移到目标模型"""
    source_dict, ckpt = load_pretrained_model(source_weights, device)
    target_dict = target_model.state_dict()
    
    # 筛选并调整权重参数
    updated_dict = {}
    unmatched_keys = []
    
    for k, v in target_dict.items():
        if k in source_dict and v.shape == source_dict[k].shape:
            # 如果形状完全匹配，直接复制权重
            updated_dict[k] = source_dict[k]
        elif k in source_dict:
            # 形状不匹配，可能是因为通道数减少
            try:
                if len(v.shape) == 4:  # 卷积核权重 [out_channels, in_channels, kernel_height, kernel_width]
                    # 根据输出通道和输入通道进行裁剪
                    updated_dict[k] = source_dict[k][:v.shape[0], :v.shape[1], :, :]
                elif len(v.shape) == 1:  # 偏置项或BatchNorm参数
                    # 只保留需要的部分
                    updated_dict[k] = source_dict[k][:v.shape[0]]
                elif len(v.shape) == 2:  # 全连接层权重
                    updated_dict[k] = source_dict[k][:v.shape[0], :v.shape[1]]
                else:
                    # 其他情况，记录为不匹配
                    unmatched_keys.append(k)
            except:
                unmatched_keys.append(k)
        else:
            unmatched_keys.append(k)
    
    # 打印不匹配的键
    if unmatched_keys:
        LOGGER.info(f"未匹配的键: {len(unmatched_keys)}/{len(target_dict)}")
        for k in unmatched_keys[:10]:  # 只显示前10个
            LOGGER.info(f"  {k}: 目标形状 {target_dict[k].shape}")
        if len(unmatched_keys) > 10:
            LOGGER.info(f"  ... 还有 {len(unmatched_keys) - 10} 个未匹配的键")
    
    # 更新权重
    target_dict.update(updated_dict)
    target_model.load_state_dict(target_dict)
    
    # 更新模型的其他信息
    new_ckpt = {
        'epoch': ckpt.get('epoch', 0),
        'best_fitness': ckpt.get('best_fitness', 0.0),
        'model': deepcopy(target_model),
        'ema': None,
        'optimizer': None,
        'training_results': ckpt.get('training_results', ''),
        'date': ckpt.get('date', ''),
        'version': ckpt.get('version', ''),
    }
    
    return new_ckpt

def main(opt):
    """主函数"""
    weights, cfg, save_path = opt.weights, opt.cfg, opt.save_path
    device = select_device(opt.device, batch_size=1)
    
    # 创建小模型
    LOGGER.info(f"创建模型: {cfg}")
    model = create_small_model(cfg, device=device)
    
    # 迁移权重
    LOGGER.info(f"加载权重: {weights}")
    new_ckpt = transfer_weights(weights, model, device)
    
    # 保存新模型
    LOGGER.info(f"保存模型到: {save_path}")
    torch.save(new_ckpt, save_path)
    LOGGER.info("迁移完成!")

def parse_opt():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--cfg', type=str, required=True, help='小模型配置文件路径')
    parser.add_argument('--save-path', type=str, required=True, help='迁移后模型保存路径')
    parser.add_argument('--device', type=str, default='', help='cuda设备, 如 0 或 0,1,2,3 或 cpu')
    
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
