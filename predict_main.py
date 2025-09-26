import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PatchTST import TrajectoryPredictor
from LSTMPredictor import LSTMTrajectoryPredictor
from sklearn.preprocessing import MinMaxScaler
import logging
import datetime
import time
from test_prediction import load_enhanced_trajectory_data, normalize_trajectories

def ensure_directory(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录：{dir_path}")

def setup_logging():
    """设置日志记录"""
    log_dir = "logs"
    ensure_directory(log_dir)
    
    # 创建日志文件名
    log_filename = os.path.join(log_dir, f"prediction_generation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # 添加控制台处理器
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return log_filename

def load_model(model_path, device):
    """加载预训练模型"""
    logging.info(f"正在加载模型：{model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logging.error(f"模型文件不存在: {model_path}")
        return None, None, None
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 提取模型参数
    input_dims = checkpoint.get('input_dims', 5)  # 默认为5维 [x, y, speed, sin(θ), cos(θ)]
    context_length = checkpoint.get('context_length', 30)
    prediction_length = checkpoint.get('prediction_length', 6)
    
    logging.info(f"模型参数: input_dims={input_dims}, context_length={context_length}, prediction_length={prediction_length}")
    
    # 初始化模型
    predictor = TrajectoryPredictor(
        input_dims=input_dims,
        context_length=context_length,
        prediction_length=prediction_length,
        patch_len=5,
        stride=2,
        d_model=128,
        n_heads=8,
        n_layers=4,
        device=device
    )
    
    # 加载预训练权重
    predictor.load_model(model_path)
    logging.info("模型加载成功！")
    
    return predictor, context_length, prediction_length

def generate_and_save_predictions(predictor, all_trajectories, scaler, context_length, prediction_length, start_step=1000):
    """生成并保存每辆车的预测轨迹"""
    output_dir = "predict_result"
    ensure_directory(output_dir)
    
    predictor.model.eval()
    num_vehicles = len(all_trajectories)
    
    logging.info(f"开始为{num_vehicles}辆车生成预测，从时间步{start_step}开始...")
    
    # 遍历每辆车
    for vehicle_id, trajectory in enumerate(all_trajectories, 1):
        logging.info(f"处理车辆 {vehicle_id}/{num_vehicles}...")
        
        # 获取轨迹长度
        traj_length = len(trajectory)
        
        # 如果轨迹长度不够，则跳过
        if traj_length <= start_step:
            logging.warning(f"车辆{vehicle_id}轨迹长度不足({traj_length})，无法从{start_step}时间步开始预测")
            continue

        # 创建结果数组：[总时间步, 预测点数, 2(x,y坐标)]
        result_shape = (traj_length, prediction_length, 2)
        predictions = np.zeros(result_shape)
        
        # 从start_step开始预测
        for t in range(start_step, traj_length - context_length + 1):
            # 计算当前预测的时间步
            current_pred_step = t
            
            # 获取历史窗口的轨迹: [t-context_length, t-1]
            history_start = current_pred_step - context_length
            history_end = current_pred_step
            history = trajectory[history_start:history_end]
            
            # 转换为tensor并添加batch维度
            history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # [1, context_length, features]
            
            # 预测
            with torch.no_grad():
                pred = predictor.predict(history_tensor)
            
            # 转为numpy
            pred_np = pred.cpu().numpy()[0]  # 移除batch维度
            
            # 如果有scaler，反归一化位置坐标
            if scaler is not None and pred_np.shape[-1] >= 2:
                # 提取位置坐标
                pos_pred = pred_np[:, :2].reshape(-1, 2)
                
                # 反归一化
                pos_pred_orig = scaler.inverse_transform(pos_pred)
                
                # 重塑回原始形状
                pos_pred = pos_pred_orig.reshape(prediction_length, 2)
                
                # 存储预测结果 - 直接存储到当前预测的时间步位置
                predictions[current_pred_step] = pos_pred
        
        # 保存预测结果到文件
        output_path = os.path.join(output_dir, f"car_{vehicle_id}_prediction.npy")
        np.save(output_path, predictions)
        logging.info(f"车辆{vehicle_id}的预测结果已保存到 {output_path}")
        
        # 可视化最后一个预测结果（用于验证）
        if current_pred_step >= start_step + 100:  # 只对预测了至少100步的车辆进行可视化
            visualize_step = current_pred_step
            
            # 获取该步的历史和预测数据
            history = trajectory[visualize_step-context_length:visualize_step]
            future = predictions[visualize_step]
            
            # 如果轨迹还够长，获取真实的未来轨迹用于比较
            true_future = None
            if visualize_step + prediction_length <= traj_length:
                true_future = trajectory[visualize_step:visualize_step+prediction_length, :2]
            
            # 可视化
            plot_path = os.path.join(output_dir, f"car_{vehicle_id}_visualization.png")
            visualize_prediction(history[:, :2], future, true_future, plot_path, vehicle_id)
    
    logging.info(f"所有车辆的预测已完成并保存到 {output_dir} 目录")
    
    return output_dir

def visualize_prediction(history, prediction, true_future=None, save_path=None, vehicle_id=None):
    """可视化单个预测结果"""
    plt.figure(figsize=(10, 6))
    
    # 绘制历史轨迹
    plt.plot(history[:, 0], history[:, 1], 'b-', linewidth=2, label='历史轨迹')
    
    # 绘制预测轨迹
    plt.plot(prediction[:, 0], prediction[:, 1], 'r--', linewidth=2, label='预测轨迹')
    
    # 绘制真实未来轨迹(如果有)
    if true_future is not None:
        plt.plot(true_future[:, 0], true_future[:, 1], 'g-', linewidth=2, label='真实未来轨迹')
    
    # 标记关键点
    plt.scatter(history[0, 0], history[0, 1], c='blue', s=100, marker='o', label='起点')
    plt.scatter(history[-1, 0], history[-1, 1], c='blue', s=100, marker='s', label='历史终点')
    plt.scatter(prediction[-1, 0], prediction[-1, 1], c='red', s=100, marker='X', label='预测终点')
    
    if true_future is not None:
        plt.scatter(true_future[-1, 0], true_future[-1, 1], c='green', s=100, marker='*', label='真实终点')
    
    plt.title(f'车辆{vehicle_id}轨迹预测', fontsize=16)
    plt.xlabel('X坐标', fontsize=14)
    plt.ylabel('Y坐标', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"可视化图像已保存至: {save_path}")
    
    plt.close()

def main():
    # 设置日志
    log_file = setup_logging()
    logging.info("开始生成车辆轨迹预测")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载最佳模型
    model_path = 'best_patchtst_trajectory_predictor.pth'
    predictor, context_length, prediction_length = load_model(model_path, device)
    
    if predictor is None:
        logging.error("模型加载失败，退出程序")
        return
    
    # 加载轨迹数据
    logging.info("正在加载车辆轨迹数据...")
    start_time = time.time()
    all_trajectories = load_enhanced_trajectory_data()
    if not all_trajectories:
        logging.error("找不到有效的轨迹数据，退出程序")
        return
    logging.info(f"已加载 {len(all_trajectories)} 辆车的轨迹数据，用时 {time.time() - start_time:.2f} 秒")
    
    # 归一化轨迹数据
    logging.info("正在归一化轨迹数据...")
    positions_only = [traj[:, :2] for traj in all_trajectories]  # 提取位置
    normalized_positions, scaler = normalize_trajectories(positions_only)
    
    # 创建完整的归一化轨迹，保留速度和方向特征
    normalized_trajectories = []
    for i, traj in enumerate(all_trajectories):
        normalized_traj = np.zeros_like(traj)
        normalized_traj[:, :2] = normalized_positions[i]  # 归一化的位置
        normalized_traj[:, 2:] = traj[:, 2:]  # 原始的速度和方向特征
        normalized_trajectories.append(normalized_traj)
    
    # 生成预测并保存结果
    start_step = 1000  # 从第1000步开始预测
    output_dir = generate_and_save_predictions(predictor, normalized_trajectories, 
                                            scaler, context_length, prediction_length, start_step)
    
    # 总结
    logging.info(f"预测结果已保存到 {output_dir} 目录")
    logging.info(f"生成的预测格式说明:")
    logging.info(f"1. 每个文件包含一辆车的全部预测结果")
    logging.info(f"2. 数据形状为 [总时间步, {prediction_length}, 2]，表示每个时间步对应的{prediction_length}个未来位置预测(x,y)")
    logging.info(f"3. 前{start_step + context_length - 1}个时间步的预测值为零，因为需要足够的历史数据才能开始预测")
    logging.info("预测任务完成！")

if __name__ == "__main__":
    main()