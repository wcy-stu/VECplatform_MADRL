import numpy as np
import random
import os
from copy import copy
import matplotlib.pyplot as plt
import time

# 全局参数
GRID_SIZE = 300  # 格点大小
MAX_GRID = 900   # 最大坐标 (3x3 网格)
SPEED_MIN = 11   # 最小速度
SPEED_MAX = 17   # 最大速度
SPEED_VARIANCE = 1.5  # 速度变化的最大幅度

# 道路网络节点(交叉路口)
INTERSECTIONS = [
    (0, 0), (300, 0), (600, 0), (900, 0),
    (0, 300), (300, 300), (600, 300), (900, 300),
    (0, 600), (300, 600), (600, 600), (900, 600),
    (0, 900), (300, 900), (600, 900), (900, 900)
]

# 车辆行为模式
VEHICLE_PATTERNS = {
    'commuter': {  # 通勤者：遵循固定路线
        'route_preference': 0.9,  # 倾向于按固定路线行驶
        'speed_stability': 0.9,   # 速度稳定性高
        'stop_probability': 0.05   # 停车概率低
    },
    'delivery': {  # 配送车辆：在特定区域内频繁移动
        'route_preference': 0.7,  # 路线相对固定
        'speed_stability': 0.7,   # 速度稳定性中等
        'stop_probability': 0.15   # 偶尔停车
    },
    'taxi': {  # 出租车：在整个区域随机移动
        'route_preference': 0.5,  # 路线变化较大
        'speed_stability': 0.6,   # 速度变化中等
        'stop_probability': 0.1    # 偶尔停车
    },
    'random': {  # 随机行驶：完全随机
        'route_preference': 0.3,  # 路线高度随机
        'speed_stability': 0.4,   # 速度变化大
        'stop_probability': 0.2    # 频繁停车
    }
}

class VehicleRouteGenerator:
    def __init__(self, vehicle_id, pattern='commuter', destination=None, times=43200):
        """
        初始化车辆路径生成器
        
        Args:
            vehicle_id: 车辆ID
            pattern: 行为模式 (commuter, delivery, taxi, random)
            destination: 目的地(如果适用)
            times: 模拟的总时间步数
        """
        self.vehicle_id = vehicle_id
        self.pattern = pattern
        self.params = VEHICLE_PATTERNS[pattern]
        self.times = times
        
        # 初始位置和方向
        self.start_point = self.random_start()
        self.current_point = self.start_point
        self.direction = 'None'
        
        # 为通勤者设置目的地
        if pattern == 'commuter':
            if destination:
                self.destination = destination
            else:
                # 随机选择一个与起点不同的交叉路口作为目的地
                possible_destinations = [i for i in INTERSECTIONS if (i[0] != self.start_point[0] or i[1] != self.start_point[1])]
                self.destination = list(random.choice(possible_destinations))
        else:
            self.destination = None
            
        # 设置偏好路线(如果是通勤者)
        if pattern == 'commuter':
            self.preferred_directions = self.calculate_preferred_directions()
        
        # 速度初始化
        self.current_speed = random.randint(SPEED_MIN, SPEED_MAX)
        
        # 上一步的行动记忆，用于增加连续性
        self.last_directions = []  # 记住最近几步的移动方向
        
    def random_start(self):
        """随机生成车辆的初始位置"""
        # 50%概率从交叉路口出发，50%概率从道路中间出发
        if random.random() < 0.5:
            # 从交叉路口出发
            return list(random.choice(INTERSECTIONS))
        else:
            # 从道路中间出发
            if random.random() < 0.5:
                # 横向道路
                x = random.randint(1, 2) * GRID_SIZE + random.randint(1, GRID_SIZE-1)
                y = random.choice([0, GRID_SIZE, 2*GRID_SIZE, 3*GRID_SIZE])
            else:
                # 纵向道路
                x = random.choice([0, GRID_SIZE, 2*GRID_SIZE, 3*GRID_SIZE])
                y = random.randint(1, 2) * GRID_SIZE + random.randint(1, GRID_SIZE-1)
            return [x, y]
    
    def calculate_preferred_directions(self):
        """为通勤者计算前往目的地的偏好方向"""
        preferred = {}
        
        # 计算水平和垂直距离以确定方向偏好
        dx = self.destination[0] - self.start_point[0]
        dy = self.destination[1] - self.start_point[1]
        
        # 设置水平方向优先级
        if dx > 0:
            preferred['horizontal'] = 'right'
        elif dx < 0:
            preferred['horizontal'] = 'left'
        else:
            preferred['horizontal'] = None
            
        # 设置垂直方向优先级
        if dy > 0:
            preferred['vertical'] = 'top'
        elif dy < 0:
            preferred['vertical'] = 'down'
        else:
            preferred['vertical'] = None
            
        return preferred
        
    def get_preferred_direction(self, current_point):
        """根据当前位置获取偏好的移动方向"""
        if self.pattern != 'commuter' or not self.destination:
            return None
            
        # 计算到目的地的距离
        dx = self.destination[0] - current_point[0]
        dy = self.destination[1] - current_point[1]
        
        # 确定优先方向
        if abs(dx) > abs(dy):
            # 横向距离更远，优先水平移动
            return 'right' if dx > 0 else 'left'
        else:
            # 纵向距离更远，优先垂直移动
            return 'top' if dy > 0 else 'down'
    
    def next_direction(self, current_point, current_direction):
        """确定下一步的移动方向"""
        # 判断当前位置是否在交叉路口
        at_intersection = (current_point[0] in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID] and 
                         current_point[1] in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID])
        
        # 1. 检查是否需要停车
        if random.random() < self.params['stop_probability']:
            return 'stop'
            
        # 2. 确定可行的方向
        available_directions = []
        
        if at_intersection:
            # 交叉路口：可以转向
            if current_point[0] > 0:
                available_directions.append('left')
            if current_point[0] < MAX_GRID:
                available_directions.append('right')
            if current_point[1] > 0:
                available_directions.append('down')
            if current_point[1] < MAX_GRID:
                available_directions.append('top')
        else:
            # 在道路上：只能直行
            if current_point[0] not in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID]:
                # 在横向道路上
                available_directions = ['left', 'right']
            else:
                # 在纵向道路上
                available_directions = ['top', 'down']
        
        # 3. 应用行为模式和偏好
        
        # 获取目的地偏好方向(如果是通勤者)
        preferred_direction = self.get_preferred_direction(current_point) if self.pattern == 'commuter' else None
        
        # 偏好保持当前方向的概率
        direction_continuity = 0.7
        
        # 如果有当前方向且该方向可行，增加其权重
        if current_direction in available_directions and current_direction != 'None' and current_direction != 'stop':
            direction_weights = [direction_continuity if d == current_direction else 
                              (1.0 - direction_continuity) / (len(available_directions) - 1) 
                              for d in available_directions]
            
            # 如果是通勤者且有偏好方向，进一步调整权重
            if self.pattern == 'commuter' and preferred_direction in available_directions:
                for i, d in enumerate(available_directions):
                    if d == preferred_direction:
                        direction_weights[i] += self.params['route_preference'] * (1.0 - direction_continuity)
                # 重新归一化权重
                direction_weights = [w / sum(direction_weights) for w in direction_weights]
                
            next_dir = random.choices(available_directions, weights=direction_weights, k=1)[0]
        else:
            # 没有当前方向或无法继续，均等选择或按偏好选择
            if self.pattern == 'commuter' and preferred_direction in available_directions:
                # 通勤者偏好朝向目的地
                direction_weights = [self.params['route_preference'] if d == preferred_direction else 
                                  (1.0 - self.params['route_preference']) / (len(available_directions) - 1)
                                  for d in available_directions]
                next_dir = random.choices(available_directions, weights=direction_weights, k=1)[0]
            else:
                # 其他模式随机选择
                next_dir = random.choice(available_directions)
        
        # 记录这个方向用于未来决策
        self.last_directions.append(next_dir)
        if len(self.last_directions) > 3:
            self.last_directions.pop(0)
            
        return next_dir
    
    def update_speed(self):
        """更新车辆速度，引入平滑变化"""
        # 速度稳定性越高，变化越小
        max_change = (1.0 - self.params['speed_stability']) * SPEED_VARIANCE
        
        # 随机变化，但变化幅度受限
        speed_change = random.uniform(-max_change, max_change)
        new_speed = self.current_speed + speed_change
        
        # 限制在合理范围内
        new_speed = max(min(new_speed, SPEED_MAX), SPEED_MIN)
        
        self.current_speed = round(new_speed, 2)
        return self.current_speed
    
    def move(self, current_point, current_speed, current_direction):
        """移动函数，决定下一个位置"""
        next_point = copy(current_point)
        next_direction = self.next_direction(current_point, current_direction)
        
        if next_direction == 'stop':
            return next_point, current_direction
            
        # 基于方向移动
        if next_direction == 'left':
            next_point[0] = max(0, next_point[0] - current_speed)
            
            # 检查是否经过交叉路口
            crossings = [x for x in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID] if 
                        current_point[0] > x and next_point[0] < x]
            if crossings:
                # 停在交叉路口
                next_point[0] = crossings[-1]
                
        elif next_direction == 'right':
            next_point[0] = min(MAX_GRID, next_point[0] + current_speed)
            
            # 检查是否经过交叉路口
            crossings = [x for x in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID] if 
                        current_point[0] < x and next_point[0] > x]
            if crossings:
                # 停在交叉路口
                next_point[0] = crossings[0]
                
        elif next_direction == 'top':
            next_point[1] = min(MAX_GRID, next_point[1] + current_speed)
            
            # 检查是否经过交叉路口
            crossings = [y for y in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID] if 
                        current_point[1] < y and next_point[1] > y]
            if crossings:
                # 停在交叉路口
                next_point[1] = crossings[0]
                
        elif next_direction == 'down':
            next_point[1] = max(0, next_point[1] - current_speed)
            
            # 检查是否经过交叉路口
            crossings = [y for y in [0, GRID_SIZE, 2*GRID_SIZE, MAX_GRID] if 
                        current_point[1] > y and next_point[1] < y]
            if crossings:
                # 停在交叉路口
                next_point[1] = crossings[-1]
                
        return next_point, next_direction
    
    def generate_route(self):
        """生成完整的车辆轨迹"""
        route = [self.current_point.copy()]  # 起始点
        speeds = [0]  # 初始速度
        directions = ['None']  # 初始方向
        
        for _ in range(self.times - 1):
            # 更新速度
            current_speed = self.update_speed()
            speeds.append(current_speed)
            
            # 移动
            next_point, next_direction = self.move(self.current_point, current_speed, self.direction)
            
            # 更新当前状态
            self.current_point = next_point.copy()
            self.direction = next_direction
            
            # 记录轨迹
            route.append(next_point.copy())
            directions.append(next_direction)
            
        return route, speeds, directions

    def save_route(self, route, speeds):
        """保存生成的轨迹"""
        # 确保目录存在
        os.makedirs('./route', exist_ok=True)
        
        # 保存轨迹和速度
        np.save(f'./route/car_{self.vehicle_id}_route.npy', route)
        np.save(f'./route/car_{self.vehicle_id}_speeds.npy', speeds)

def visualize_routes(vehicle_routes, vehicle_ids=None, save_path=None):
    """可视化多条车辆轨迹"""
    plt.figure(figsize=(12, 12))
    
    # 绘制道路网格
    for i in range(0, MAX_GRID + GRID_SIZE, GRID_SIZE):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)
    
    # 绘制车辆轨迹
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'olive']
    
    for i, (route, vehicle_id) in enumerate(zip(vehicle_routes, vehicle_ids or range(len(vehicle_routes)))):
        color = colors[i % len(colors)]
        
        # 将路线转为numpy数组以便于操作
        route_array = np.array(route)
        
        # 绘制路线
        plt.plot(route_array[:, 0], route_array[:, 1], color=color, alpha=0.7, label=f'Vehicle {vehicle_id}')
        
        # 标记起点和终点
        plt.scatter(route_array[0, 0], route_array[0, 1], color=color, s=100, marker='o')
        plt.scatter(route_array[-1, 0], route_array[-1, 1], color=color, s=100, marker='x')
    
    plt.title('Vehicle Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def generate_vehicle_routes(num_vehicles=12, times=43200):
    """生成多辆车的轨迹并保存"""
    all_routes = []
    vehicle_ids = []
    
    # 定义车辆行为模式分布
    pattern_distribution = {
        'commuter': 0.5,  # 50%车辆是通勤者
        'delivery': 0.3,  # 30%车辆是配送车辆
        'taxi': 0.15,     # 15%车辆是出租车
        'random': 0.05    # 5%车辆是随机行驶
    }
    
    for i in range(1, num_vehicles + 1):
        print(f"生成车辆{i}的轨迹...")
        
        # 随机选择行为模式
        pattern = random.choices(
            list(pattern_distribution.keys()),
            weights=list(pattern_distribution.values()),
            k=1
        )[0]
        
        generator = VehicleRouteGenerator(i, pattern=pattern, times=times)
        route, speeds, _ = generator.generate_route()
        generator.save_route(route, speeds)
        
        all_routes.append(route)
        vehicle_ids.append(i)
        
        # 创建任务
        x = create_save_task('car_' + str(i), 0, times)
        
        time.sleep(1)  # 稍作延迟以确保随机数生成的差异
    
    # 可视化所有车辆轨迹
    visualize_routes(all_routes, vehicle_ids, save_path='./route/all_vehicles_trajectories.png')
    
    # 可视化前3辆车的轨迹(更清晰)
    visualize_routes(all_routes[:3], vehicle_ids[:3], save_path='./route/sample_vehicles_trajectories.png')
    
    return all_routes

# 从utils.py导入需要的函数
from utils import create_save_task

if __name__ == "__main__":
    # 生成12辆车的轨迹，每辆车12小时(43200秒)
    generate_vehicle_routes(num_vehicles=12, times=43200)