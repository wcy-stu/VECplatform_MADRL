import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PatchTST import TrajectoryPredictor
from LSTMPredictor import LSTMTrajectoryPredictor  # 导入LSTM模型
import os
import time
import argparse  # 用于命令行参数解析
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from physical_distance_loss import PhysicalDistanceLoss
import logging
import datetime

# 创建日志文件夹函数
def ensure_log_directory(log_dir="logs"):
    """确保日志目录存在"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

class TrajectoryDataset(Dataset):
    """车辆轨迹数据集"""
    
    def __init__(self, trajectories, context_length=20, prediction_length=10):
        self.data_pairs = []
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # 为每条轨迹创建输入-输出对
        for traj in trajectories:
            # 检查轨迹是否足够长
            if len(traj) < context_length + prediction_length:
                continue
                
            # 使用滑动窗口创建多个样本
            for i in range(0, len(traj) - context_length - prediction_length + 1, 5):  # 每5步采样一个样本，减少数据量
                x = traj[i:i+context_length]
                y = traj[i+context_length:i+context_length+prediction_length]
                self.data_pairs.append((x, y))
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        x, y = self.data_pairs[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_trajectory_data(num_vehicles=12):
    """加载所有车辆的轨迹数据"""
    all_trajectories = []
    
    for i in range(1, num_vehicles + 1):
        try:
            # 加载车辆轨迹
            file_path = f'route/car_{i}_route.npy'
            if os.path.exists(file_path):
                trajectory = np.load(file_path, allow_pickle=True)
                all_trajectories.append(trajectory)
                print(f"已加载车辆{i}的轨迹，形状: {trajectory.shape}")
            else:
                print(f"找不到车辆{i}的轨迹文件")
        except Exception as e:
            print(f"加载车辆{i}的轨迹时出错: {e}")
    
    return all_trajectories

# 加载位置和速度数据
def load_enhanced_trajectory_data(num_vehicles=12):
    all_trajectories = []
    
    for i in range(1, num_vehicles + 1):
        try:
            # 加载位置轨迹
            route_path = f'route/car_{i}_route.npy'
            speed_path = f'route/car_{i}_speeds.npy'
            
            if os.path.exists(route_path) and os.path.exists(speed_path):
                positions = np.load(route_path, allow_pickle=True)
                speeds = np.load(speed_path, allow_pickle=True)
                
                # 计算方向（角度）
                directions = np.zeros((len(positions), 2))  # [sin(θ), cos(θ)]
                for j in range(1, len(positions)):
                    dx = positions[j][0] - positions[j-1][0]
                    dy = positions[j][1] - positions[j-1][1]
                    if dx != 0 or dy != 0:  # 避免计算静止点的方向
                        angle = math.atan2(dy, dx)
                        directions[j][0] = math.sin(angle)
                        directions[j][1] = math.cos(angle)
                    else:
                        directions[j] = directions[j-1]  # 保持前一个方向
                
                # 首个点的方向与第二个相同
                directions[0] = directions[1]
                
                # 将所有特征合并为一个数组 [x, y, speed, sin(θ), cos(θ)]
                speeds_reshaped = speeds.reshape(-1, 1)
                enhanced_trajectory = np.concatenate([positions, speeds_reshaped, directions], axis=1)
                
                all_trajectories.append(enhanced_trajectory)
                print(f"已加载车辆{i}的增强轨迹，形状: {enhanced_trajectory.shape}")
            else:
                print(f"找不到车辆{i}的轨迹文件或速度文件")
        except Exception as e:
            print(f"加载车辆{i}的轨迹时出错: {e}")
    
    return all_trajectories

def normalize_trajectories(trajectories):
    """归一化所有轨迹数据"""
    # 找出所有坐标的最大最小值
    all_coords = np.vstack(trajectories)
    scaler = MinMaxScaler()
    scaler.fit(all_coords)
    
    # 对每条轨迹进行归一化
    normalized = []
    for traj in trajectories:
        norm_traj = scaler.transform(traj)
        normalized.append(norm_traj)
    
    return normalized, scaler

def evaluate_predictions(predictor, test_loader, device, scaler=None):
    """评估模型预测效果"""
    predictor.model.eval()
    all_mae = []
    all_mse = []
    distance_errors = []
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x, y_true = x.to(device), y_true.to(device)
            
            # 预测
            y_pred = predictor.predict(x)
            
            # 计算MAE和MSE - 只考虑位置坐标 (前两个维度)
            if y_true.shape[-1] > 2 and y_pred.shape[-1] > 2:
                mae = torch.abs(y_pred[..., :2] - y_true[..., :2]).mean(dim=(1, 2))
                mse = ((y_pred[..., :2] - y_true[..., :2]) ** 2).mean(dim=(1, 2))
            else:
                mae = torch.abs(y_pred - y_true).mean(dim=(1, 2))
                mse = ((y_pred - y_true) ** 2).mean(dim=(1, 2))
            
            # 如果有缩放器，将坐标转换回原始空间计算实际距离误差
            if scaler:
                # 转换为numpy以便使用scaler
                y_pred_np = y_pred.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                
                # 重塑为2D以便使用scaler - 只使用位置坐标
                batch_size = y_pred_np.shape[0]
                pred_len = y_pred_np.shape[1]
                
                # 只提取位置坐标（前两个维度）
                if y_pred_np.shape[-1] > 2:
                    y_pred_flat = y_pred_np[:,:,:2].reshape(-1, 2)
                    y_true_flat = y_true_np[:,:,:2].reshape(-1, 2)
                else:
                    y_pred_flat = y_pred_np.reshape(-1, 2)
                    y_true_flat = y_true_np.reshape(-1, 2)
                
                # 反归一化
                y_pred_orig = scaler.inverse_transform(y_pred_flat)
                y_true_orig = scaler.inverse_transform(y_true_flat)
                
                # 重塑回3D
                y_pred_orig = y_pred_orig.reshape(batch_size, pred_len, 2)
                y_true_orig = y_true_orig.reshape(batch_size, pred_len, 2)
                
                # 计算欧氏距离误差
                for i in range(batch_size):
                    for j in range(pred_len):
                        dist = np.sqrt(np.sum((y_pred_orig[i, j] - y_true_orig[i, j]) ** 2))
                        distance_errors.append(dist)
            
            all_mae.extend(mae.cpu().numpy())
            all_mse.extend(mse.cpu().numpy())
    
    # 计算准确率 (定义为预测点与真实点距离小于阈值的百分比)
    thresholds = [5, 10, 20, 50]  # 阈值单位：米
    accuracies = {}
    
    if distance_errors:
        for threshold in thresholds:
            accuracy = np.mean([1 if err < threshold else 0 for err in distance_errors])
            accuracies[threshold] = accuracy
    
    # 计算平均指标
    avg_mae = np.mean(all_mae)
    avg_mse = np.mean(all_mse)
    
    results = {
        'avg_mae': avg_mae,
        'avg_mse': avg_mse,
        'rmse': np.sqrt(avg_mse),
        'accuracy': accuracies,
        'distance_errors': distance_errors
    }
    
    return results

def visualize_predictions(predictor, test_loader, device, num_samples=3, scaler=None, save_prefix="trajectory_prediction_sample"):
    """可视化预测结果"""
    predictor.model.eval()
    samples_plotted = 0
    
    with torch.no_grad():
        for x, y_true in test_loader:
            if samples_plotted >= num_samples:
                break
                
            x, y_true = x.to(device), y_true.to(device)
            y_pred = predictor.predict(x)
            
            # 转为CPU并获取numpy数组
            x_np = x.cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            # 如果有缩放器，转换回原始坐标
            if scaler:
                # 提取位置坐标
                if x_np.shape[-1] > 2:
                    x_pos_np = x_np[:,:,:2]
                    y_true_pos_np = y_true_np[:,:,:2]
                    y_pred_pos_np = y_pred_np[:,:,:2]
                else:
                    x_pos_np = x_np
                    y_true_pos_np = y_true_np
                    y_pred_pos_np = y_pred_np
                
                # 反归一化历史轨迹
                x_flat = x_pos_np.reshape(-1, 2)
                x_orig = scaler.inverse_transform(x_flat)
                x_pos_np = x_orig.reshape(x_pos_np.shape)
                
                # 反归一化真实未来轨迹
                y_true_flat = y_true_pos_np.reshape(-1, 2)
                y_true_orig = scaler.inverse_transform(y_true_flat)
                y_true_pos_np = y_true_orig.reshape(y_true_pos_np.shape)
                
                # 反归一化预测未来轨迹
                y_pred_flat = y_pred_pos_np.reshape(-1, 2)
                y_pred_orig = scaler.inverse_transform(y_pred_flat)
                y_pred_pos_np = y_pred_orig.reshape(y_pred_pos_np.shape)
            else:
                # 如果没有缩放器，但仍需要提取位置坐标
                if x_np.shape[-1] > 2:
                    x_pos_np = x_np[:,:,:2]
                    y_true_pos_np = y_true_np[:,:,:2]
                    y_pred_pos_np = y_pred_np[:,:,:2]
                else:
                    x_pos_np = x_np
                    y_true_pos_np = y_true_np
                    y_pred_pos_np = y_pred_np
            
            # 为每个样本绘制图像
            for i in range(min(num_samples - samples_plotted, x_pos_np.shape[0])):
                plt.figure(figsize=(10, 6))
                
                # 绘制历史轨迹
                plt.plot(x_pos_np[i, :, 0], x_pos_np[i, :, 1], 'b-', label='历史轨迹')
                
                # 绘制真实未来轨迹
                plt.plot(y_true_pos_np[i, :, 0], y_true_pos_np[i, :, 1], 'g-', label='真实未来轨迹')
                
                # 绘制预测未来轨迹
                plt.plot(y_pred_pos_np[i, :, 0], y_pred_pos_np[i, :, 1], 'r--', label='预测未来轨迹')
                
                # 标记起点和终点
                plt.scatter(x_pos_np[i, 0, 0], x_pos_np[i, 0, 1], c='blue', s=50, marker='o', label='起点')
                plt.scatter(x_pos_np[i, -1, 0], x_pos_np[i, -1, 1], c='blue', s=50, marker='s', label='历史终点')
                plt.scatter(y_true_pos_np[i, -1, 0], y_true_pos_np[i, -1, 1], c='green', s=50, marker='*', label='真实终点')
                plt.scatter(y_pred_pos_np[i, -1, 0], y_pred_pos_np[i, -1, 1], c='red', s=50, marker='X', label='预测终点')
                
                # 添加图例和标签
                plt.legend(loc='upper right')
                plt.title('车辆轨迹预测可视化')
                plt.xlabel('X坐标')
                plt.ylabel('Y坐标')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图像
                plt.savefig(f'{save_prefix}_{samples_plotted+i+1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            samples_plotted += x_pos_np.shape[0]

def test_trajectory_prediction(model_type='patchtst'):
    """测试轨迹预测模型的主函数"""
    
    # 清理之前的所有日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 确保日志目录存在
    log_dir = ensure_log_directory()
    
    # 设置日志文件路径
    log_filename = os.path.join(log_dir, f"{model_type}_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # 覆盖模式
    )
    
    # 同时输出到控制台和文件
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"=== 开始轨迹预测测试 (模型: {model_type}) ===")
    logging.info(f"日志文件保存在: {log_filename}")
    
    # 设置参数
    context_length = 30        # 历史长度
    prediction_length = 6      # 预测长度
    batch_size = 32
    num_epochs = 100
    test_size = 0.2            # 测试集比例 
    
    # 加载轨迹数据 - 加载增强轨迹
    logging.info("加载增强轨迹数据...")
    all_trajectories = load_enhanced_trajectory_data()
    
    if not all_trajectories:
        logging.info("没有找到有效的轨迹数据！")
        return None
        
    logging.info(f"共加载了 {len(all_trajectories)} 辆车的轨迹数据")
    
    # 归一化轨迹 - 我们只对位置进行归一化
    logging.info("归一化轨迹数据...")
    positions_only = [traj[:, :2] for traj in all_trajectories]  # 提取位置
    normalized_positions, scaler = normalize_trajectories(positions_only)
    
    # 创建完整的归一化轨迹，保留速度和方向特征
    normalized_trajectories = []
    for i, traj in enumerate(all_trajectories):
        # 创建一个新的数组，复制归一化后的位置
        normalized_traj = np.zeros_like(traj)
        normalized_traj[:, :2] = normalized_positions[i]  # 归一化的位置
        normalized_traj[:, 2:] = traj[:, 2:]  # 原始的速度和方向特征
        normalized_trajectories.append(normalized_traj)
    
    # 基于时间序列进行分割
    logging.info("准备数据集...")
    train_trajectories = []
    test_trajectories = []
    
    for traj in normalized_trajectories:
        # 使用前40000个时间步作为训练数据（如果轨迹不够长则使用80%）
        split_point = min(40000, int(len(traj) * 0.8))
        if split_point > context_length + prediction_length:  # 确保训练集有足够数据
            train_trajectories.append(traj[:split_point])
            if len(traj) > split_point + context_length:  # 确保测试集有足够数据
                test_trajectories.append(traj[split_point:])
    
    # 创建数据集和数据加载器
    train_dataset = TrajectoryDataset(train_trajectories, context_length, prediction_length)
    test_dataset = TrajectoryDataset(test_trajectories, context_length, prediction_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logging.info(f"训练集包含 {len(train_dataset)} 个样本")
    logging.info(f"测试集包含 {len(test_dataset)} 个样本")
    
    # 检查是否有足够的数据
    if len(train_dataset) < batch_size or len(test_dataset) < batch_size:
        logging.info("警告：数据样本不足，请增加轨迹数据或减少批次大小！")
        batch_size = min(len(train_dataset), len(test_dataset))
        if batch_size == 0:
            logging.info("没有足够的数据进行训练和测试！")
            return None
        logging.info(f"调整批次大小为 {batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型 - 根据指定类型选择模型
    logging.info(f"初始化{model_type.upper()}模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    input_dims = 5  # [x, y, speed, sin(θ), cos(θ)]
    
    if model_type.lower() == 'patchtst':
        # 使用增强型PatchTST模型
        predictor = TrajectoryPredictor(
            input_dims=input_dims,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_len=5,
            stride=2,
            d_model=128,      # 使用128维特征
            n_heads=8,        # 8个注意力头
            n_layers=4,       # 4层transformer
            dropout=0.1,
            fc_dropout=0.1,
            learning_rate=0.001,  # 初始学习率降低
            device=device
        )
        model_file = 'patchtst_trajectory_predictor.pth'
    else:
        # 使用改进的LSTM模型
        predictor = LSTMTrajectoryPredictor(
            input_dims=input_dims,
            hidden_dims=256,            # 增大隐藏层维度
            output_dims=input_dims,
            context_length=context_length,
            prediction_length=prediction_length,
            num_layers=3,              # 增加层数
            dropout=0.2,               # 增加dropout
            bidirectional=False,       # 单向LSTM更适合预测任务
            learning_rate=0.003,       # 调整学习率
            use_attention=True,        # 使用注意力机制
            device=device
        )
        model_file = 'lstm_trajectory_predictor.pth'
    
    # 替换损失函数为物理距离损失，调整权重
    predictor.loss_fn = PhysicalDistanceLoss(
        scaler=scaler,
        position_weight=5.0,    # 降低位置坐标MSE权重
        aux_weight=0.5,         # 为辅助特征添加小权重
        distance_weight=3.0    # 增加物理距离权重
    )
    
    # 修改早停参数和逻辑
    if model_type.lower() == 'lstm':
        # LSTM模型需要更多耐心和更宽松的改进标准
        max_patience = 30  # 增加LSTM的耐心值
        improvement_threshold = 0.002  # 添加改进阈值，只要有微小改进就重置耐心计数
    else:
        max_patience = 15  # PatchTST保持原来的耐心值
        improvement_threshold = 0.001  # 更严格的改进要求
    
    # 训练模型
    logging.info("开始训练模型...")
    start_time = time.time()
    
    # 添加学习率调度
    if model_type.lower() == 'lstm':
        # 为LSTM使用OneCycleLR调度器
        from torch.optim.lr_scheduler import OneCycleLR
        total_steps = num_epochs * len(train_loader)
        scheduler = OneCycleLR(
            predictor.optimizer,
            max_lr=0.01,
            total_steps=total_steps,
            pct_start=0.1,         # 10%时间用于预热
            div_factor=25,         # 初始学习率 = max_lr/25
            final_div_factor=1000,  # 最终学习率 = max_lr/1000
            anneal_strategy='cos'  # 余弦退火
        )
    else:
        # 其他模型保持原来的ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(predictor.optimizer, 'min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience_counter = 0
    
    predictor.model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            predictor.optimizer.zero_grad()
            
            # 对LSTM使用教师强制
            if model_type.lower() == 'lstm':
                y_pred = predictor.model(x, target=y, teacher_forcing_ratio=0.5)  # 50%几率使用教师强制
            else:
                y_pred = predictor.model(x)
                
            loss = predictor.loss_fn(y_pred, y)
            loss.backward()
            
            # 梯度裁剪 (对LSTM特别重要)
            if model_type.lower() == 'lstm':
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
                
            predictor.optimizer.step()
            
            # 对OneCycleLR每批次更新学习率
            if model_type.lower() == 'lstm' and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # ReduceLROnPlateau每轮更新学习率
        if model_type.lower() != 'lstm' or isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_loss)
        
        # 在训练循环中修改早停检查
        if avg_loss < best_loss * (1 - improvement_threshold):  # 至少改进了 0.X%
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            predictor.save_model(f'best_{model_file}')
            logging.info(f"Epoch {epoch + 1}: 发现更好的模型，损失从 {best_loss:.6f} 改善到 {avg_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"早停：{max_patience}轮无明显改善 (当前最佳: {best_loss:.6f}, 当前: {avg_loss:.6f})")
                break
                
        if (epoch + 1) % 10 == 0:
            log_message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {predictor.optimizer.param_groups[0]['lr']:.6f}"
            logging.info(log_message)
    
    training_time = time.time() - start_time
    logging.info(f"模型训练完成，耗时 {training_time:.2f} 秒")
    
    # 保存模型
    predictor.save_model(model_file)
    logging.info(f"模型已保存至 {model_file}")
    
    # 评估模型
    logging.info("评估模型性能...")
    eval_results = evaluate_predictions(predictor, test_loader, device, scaler)
    
    # 输出评估结果
    logging.info(f"\n=== {model_type.upper()} 预测性能评估 ===")
    logging.info(f"平均绝对误差 (MAE): {eval_results['avg_mae']:.4f}")
    logging.info(f"均方误差 (MSE): {eval_results['avg_mse']:.4f}")
    logging.info(f"均方根误差 (RMSE): {eval_results['rmse']:.4f}")
    
    if 'accuracy' in eval_results:
        logging.info("\n预测准确率 (距离误差小于阈值的比例):")
        for threshold, accuracy in eval_results['accuracy'].items():
            logging.info(f"  - 阈值 {threshold}米: {accuracy*100:.2f}%")
    
    # 计算平均距离误差
    if 'distance_errors' in eval_results and eval_results['distance_errors']:
        distance_errors = eval_results['distance_errors']
        avg_distance = np.mean(distance_errors)
        median_distance = np.median(distance_errors)
        max_distance = np.max(distance_errors)
        logging.info(f"\n距离误差统计 (米):")
        logging.info(f"  - 平均: {avg_distance:.2f}")
        logging.info(f"  - 中位数: {median_distance:.2f}")
        logging.info(f"  - 最大: {max_distance:.2f}")
    
    # 可视化一些预测结果
    logging.info("\n生成预测可视化图...")
    visualization_folder = f"{model_type}_visualizations"
    os.makedirs(visualization_folder, exist_ok=True)
    visualize_predictions(predictor, test_loader, device, num_samples=5, scaler=scaler, 
                          save_prefix=f"{visualization_folder}/{model_type}_pred")
    logging.info(f"可视化图已生成并保存到 {visualization_folder} 文件夹")
    
    logging.info(f"\n=== {model_type.upper()} 轨迹预测测试完成 ===")
    
    # 返回评估结果 - 这是关键步骤，确保结果能被compare_models使用
    return eval_results

def compare_models():
    """比较不同模型的性能"""
    # 清理之前的所有日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 确保日志目录存在
    log_dir = ensure_log_directory()
    
    # 设置日志文件路径
    log_filename = os.path.join(log_dir, f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # 同时输出到控制台和文件
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    # 保存原始print函数，以便后续恢复
    original_print = print
    
    try:
        logging.info("=== 模型性能对比测试 ===")
        logging.info(f"日志文件保存在: {log_filename}")
        
        # 测试PatchTST模型
        logging.info("\n[1/2] 测试PatchTST模型...")
        patchtst_results = test_trajectory_prediction(model_type='patchtst')
        
        # 清理日志处理器，为下一个模型测试做准备
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 恢复原始print函数
        import builtins
        builtins.print = original_print
        
        # 测试LSTM模型
        logging.info("\n[2/2] 测试LSTM模型...")
        lstm_results = test_trajectory_prediction(model_type='lstm')
        
        # 比较结果记录到日志
        log_comparison_results(patchtst_results, lstm_results)
        
    finally:
        # 确保无论如何都恢复原始print函数
        import builtins
        builtins.print = original_print
    
    return patchtst_results, lstm_results


def log_comparison_results(patchtst_results, lstm_results):
    """记录比较结果到日志"""
    logging.info("\n=== 模型性能对比 ===")
    metrics = ['avg_mae', 'avg_mse', 'rmse']
    
    logging.info(f"{'指标':<10} {'PatchTST':<15} {'LSTM':<15} {'差异 (LSTM-PatchTST)':<20}")
    logging.info("-" * 60)
    
    for metric in metrics:
        patchtst_val = patchtst_results[metric]
        lstm_val = lstm_results[metric]
        diff = lstm_val - patchtst_val
        logging.info(f"{metric:<10} {patchtst_val:.4f} {' '*8} {lstm_val:.4f} {' '*8} {diff:.4f}")
    
    # 比较准确率
    logging.info("\n准确率对比 (距离误差小于阈值的比例):")
    for threshold in [5, 10, 20, 50]:
        patchtst_acc = patchtst_results['accuracy'][threshold]
        lstm_acc = lstm_results['accuracy'][threshold]
        diff = lstm_acc - patchtst_acc
        logging.info(f"阈值 {threshold}米: PatchTST: {patchtst_acc*100:.2f}%, LSTM: {lstm_acc*100:.2f}%, 差异: {diff*100:.2f}%")
    
    # 比较距离误差
    logging.info("\n距离误差对比 (米):")
    patchtst_dist = np.mean(patchtst_results['distance_errors'])
    lstm_dist = np.mean(lstm_results['distance_errors'])
    logging.info(f"平均距离误差: PatchTST: {patchtst_dist:.2f}米, LSTM: {lstm_dist:.2f}米, 差异: {lstm_dist-patchtst_dist:.2f}米")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='车辆轨迹预测测试')
    parser.add_argument('--model', type=str, choices=['patchtst', 'lstm', 'compare'], default='compare',
                        help='选择使用的模型: patchtst, lstm 或 compare (比较两者)')
    args = parser.parse_args()
    
    if args.model == 'compare':
        compare_models()
    else:
        test_trajectory_prediction(model_type=args.model)