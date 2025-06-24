import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5, beta=0.5, gamma=0.5):
        """
        初始化知识蒸馏损失
        
        参数:
            teacher_model: 教师模型（大模型）
            student_model: 学生模型（小模型）
            temperature: 软标签的温度参数，较高的温度会产生更柔和的概率分布
            alpha: 特征蒸馏损失的权重系数
            beta: 分类蒸馏损失的权重系数
            gamma: 边界框回归蒸馏损失的权重系数
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 设置教师模型为评估模式
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def feature_distillation_loss(self, teacher_features, student_features):
        """特征蒸馏损失，用于中间层特征的蒸馏"""
        loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            # 如果特征图尺寸不同，将学生特征插值到与教师特征相同的尺寸
            if t_feat.shape != s_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # 计算特征之间的MSE损失
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss

    def logit_distillation_loss(self, teacher_logits, student_logits):
        """分类蒸馏损失，使用KL散度计算软标签之间的差异"""
        t_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
        s_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # 计算KL散度损失
        loss = F.kl_div(s_prob, t_prob, reduction='batchmean') * (self.temperature ** 2)
        return loss

    def bbox_distillation_loss(self, teacher_bbox, student_bbox):
        """边界框回归蒸馏损失"""
        # 使用smooth L1损失进行边界框回归蒸馏
        return F.smooth_l1_loss(student_bbox, teacher_bbox)

    def get_distillation_loss(self, student_outputs, teacher_outputs):
        """
        计算总蒸馏损失
        
        参数:
            student_outputs: 学生模型的输出
            teacher_outputs: 教师模型的输出
            
        返回:
            总蒸馏损失
        """
        # 提取特征、分类和边界框输出
        t_features, t_cls_logits, t_bbox = teacher_outputs
        s_features, s_cls_logits, s_bbox = student_outputs
        
        # 计算特征蒸馏损失
        feat_loss = self.feature_distillation_loss(t_features, s_features)
        
        # 计算分类蒸馏损失
        cls_loss = self.logit_distillation_loss(t_cls_logits, s_cls_logits)
        
        # 计算边界框回归蒸馏损失
        bbox_loss = self.bbox_distillation_loss(t_bbox, s_bbox)
        
        # 计算总损失
        total_loss = self.alpha * feat_loss + self.beta * cls_loss + self.gamma * bbox_loss
        
        return total_loss, {
            'feature_loss': feat_loss.item(),
            'classification_loss': cls_loss.item(),
            'bbox_loss': bbox_loss.item(),
        }

