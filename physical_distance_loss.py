import torch
import torch.nn as nn
import numpy as np

class PhysicalDistanceLoss(nn.Module):
    """结合物理距离的损失函数，同时优化归一化空间MSE和实际物理距离"""
    
    def __init__(self, scaler, position_weight=1.0, aux_weight=0.1, distance_weight=2.0):
        """
        初始化物理距离损失函数
        
        参数:
            scaler: 用于坐标反归一化的MinMaxScaler对象
            position_weight: 位置坐标MSE损失的权重 (x,y)
            aux_weight: 辅助特征MSE损失的权重 (speed,sin,cos)
            distance_weight: 物理欧氏距离损失的权重
        """
        super(PhysicalDistanceLoss, self).__init__()
        self.scaler = scaler
        self.position_weight = position_weight
        self.aux_weight = aux_weight
        self.distance_weight = distance_weight
    
    def forward(self, y_pred, y_true):
        """计算综合损失"""
        batch_size, seq_len, feat_dims = y_pred.shape
        
        # 1. 计算位置坐标的MSE损失
        pos_loss = ((y_pred[..., :2] - y_true[..., :2]) ** 2).mean() * self.position_weight
        
        # 2. 计算辅助特征的MSE损失(如果有)
        if feat_dims > 2:
            aux_loss = ((y_pred[..., 2:] - y_true[..., 2:]) ** 2).mean() * self.aux_weight
        else:
            aux_loss = 0
        
        # 3. 计算物理距离损失
        try:
            # 提取位置坐标
            pos_pred = y_pred[..., :2].detach().cpu().numpy().reshape(-1, 2)
            pos_true = y_true[..., :2].detach().cpu().numpy().reshape(-1, 2)
            
            # 反归一化
            pos_pred_orig = self.scaler.inverse_transform(pos_pred).reshape(batch_size, seq_len, 2)
            pos_true_orig = self.scaler.inverse_transform(pos_true).reshape(batch_size, seq_len, 2)
            
            # 计算欧氏距离
            dists = np.sqrt(np.sum((pos_pred_orig - pos_true_orig) ** 2, axis=2))
            physical_loss = torch.tensor(np.mean(dists), device=y_pred.device) * self.distance_weight / 100.0
            
        except Exception as e:
            print(f"计算物理距离时出错: {e}")
            physical_loss = torch.tensor(0.0, device=y_pred.device)
        
        # 总损失
        total_loss = pos_loss + aux_loss + physical_loss
        return total_loss