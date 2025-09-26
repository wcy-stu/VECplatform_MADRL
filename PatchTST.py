import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
import os
import sys

# 添加PatchTST源码路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PatchTST_test/PatchTST_supervised'))

# 从PatchTST源码导入必要组件
from layers.PatchTST_backbone import PatchTST_backbone
from layers.RevIN import RevIN

class EnhancedPatchTST(nn.Module):
    """
    增强版PatchTST模型，基于原始PatchTST_backbone实现
    """
    def __init__(
        self,
        input_dims=2,           # 输入维度 (x,y坐标等)
        context_length=20,      # 历史序列长度
        prediction_length=10,   # 预测序列长度
        patch_len=5,            # 每个补丁的长度
        stride=2,               # 补丁之间的步长
        d_model=128,            # transformer维度
        n_heads=8,              # 注意力头数
        n_layers=4,             # transformer层数
        dropout=0.1,            # dropout率
        fc_dropout=0.1,         # 最终fc层的dropout率
        head_type='flatten',    # 头部类型
        revin=True,             # 是否使用RevIN
        affine=True,            # RevIN是否使用仿射变换
        subtract_last=False,    # RevIN是否减去最后一个值
        individual=False,       # 是否使用独立的头部
    ):
        super(EnhancedPatchTST, self).__init__()
        
        self.input_dims = input_dims
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # 使用PatchTST_backbone作为模型主体
        self.model = PatchTST_backbone(
            c_in=input_dims,                   # 输入通道数
            context_window=context_length,     # 上下文窗口
            target_window=prediction_length,   # 目标预测长度
            patch_len=patch_len,               # 补丁长度
            stride=stride,                     # 补丁步长
            n_layers=n_layers,                 # Transformer层数
            d_model=d_model,                   # 模型维度
            n_heads=n_heads,                   # 注意力头数
            d_ff=d_model*4,                    # 前馈网络维度
            dropout=dropout,                   # dropout率
            fc_dropout=fc_dropout,             # 全连接dropout率
            head_dropout=0.0,                  # 头部dropout率
            head_type=head_type,               # 头部类型
            individual=individual,             # 是否使用独立头部
            revin=revin,                       # 是否使用RevIN
            affine=affine,                     # RevIN是否使用仿射
            subtract_last=subtract_last,       # RevIN是否减去最后值
            act='gelu',                        # 激活函数
            norm='BatchNorm',                  # 归一化类型
        )
    
    def forward(self, x):
        # 输入形状: [batch_size, context_length, input_dims]
        
        # PatchTST_backbone期望输入形状为[batch_size, input_dims, context_length]
        x = x.permute(0, 2, 1)
        
        # 通过模型前向传播
        output = self.model(x)
        
        # 输出形状: [batch_size, input_dims, prediction_length]
        # 转换回 [batch_size, prediction_length, input_dims]
        output = output.permute(0, 2, 1)
        
        return output


class TrajectoryPredictor:
    """
    车辆轨迹预测器类，使用增强型PatchTST作为底层模型
    """
    def __init__(
        self,
        input_dims=5,           # 轨迹特征维度 [x, y, speed, sin(θ), cos(θ)]
        context_length=16,      # 历史序列长度
        prediction_length=6,    # 预测序列长度
        patch_len=5,            # 补丁长度
        stride=2,               # 补丁间隔
        d_model=128,            # 模型维度
        n_heads=8,              # 注意力头数
        n_layers=4,             # Transformer层数
        dropout=0.1,            # dropout率
        fc_dropout=0.1,         # 最终fc层的dropout率
        learning_rate=0.001,    # 学习率
        device=None             # 计算设备
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 初始化增强型PatchTST模型
        self.model = EnhancedPatchTST(
            input_dims=input_dims,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            fc_dropout=fc_dropout,
            revin=True,            # 启用RevIN以提高性能
            head_type='flatten',   # 使用flatten头部
        ).to(self.device)
        
        # 使用AdamW优化器而不是Adam，包含权重衰减
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01     # 添加L2正则化
        )
        
        # 初始化为MSE损失，可以后续被替换为物理距离损失
        self.loss_fn = nn.MSELoss()
        
        # 存储参数
        self.input_dims = input_dims
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def train(self, train_loader, num_epochs=100, scheduler=None, verbose=True):
        """使用数据加载器训练模型"""
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                # 更新学习率调度器
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                        
                total_loss += loss.item()
            
            # 每轮结束后更新ReduceLROnPlateau类型的调度器
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss / len(train_loader))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def predict(self, trajectory_history):
        """预测未来轨迹"""
        self.model.eval()
        with torch.no_grad():
            x = trajectory_history.to(self.device)
            prediction = self.model(x)
        return prediction
    
    def _prepare_dataset(self, trajectories):
        """准备训练数据集"""
        dataset = []
        for traj in trajectories:
            if len(traj) < self.context_length + self.prediction_length:
                continue
                
            for i in range(0, len(traj) - self.context_length - self.prediction_length + 1, 5):
                x = traj[i:i+self.context_length]
                y = traj[i+self.context_length:i+self.context_length+self.prediction_length]
                
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float32)
                
                dataset.append((x, y))
        
        return dataset
    
    def save_model(self, path):
        """保存模型到指定路径"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dims': self.input_dims,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length
        }, path)
    
    def load_model(self, path):
        """从指定路径加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])