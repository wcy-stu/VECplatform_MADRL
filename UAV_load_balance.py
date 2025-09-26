import math
import os
import random

import numpy as np
from torch import optim

import config
import model
import torch
import utils
import torch.nn.functional as F
from torch.distributions import Categorical
# from comet_ml import Experiment

class UAV:
    def __init__(self, id, pos, mode, cloud, dictionary, buffer, sfns=None):
        self.global_workload = True
        self.predict = dictionary["prediction"]
        self.id = id
        self.pos = pos
        self.mode = mode
        self.f = config.f_uav[id]
        self.bind_car = list()
        self.mode = mode
        self.cloud = cloud
        self.sfns = sfns
        self.send_cloud_queue = list()
        self.compute_queue = list()
        self.task_success = 0
        self.task_res = 0
        self.task_drop = 0
        self.time_counter = 0
        # 添加回合奖励跟踪
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_task_count = 0
        self.episode_length = 50  # 每50个任务为一个回合
        self.episode_number = 0

        # 添加任务追踪字典，用于关联任务ID和奖励
        self.task_rewards = {}

        self.n_state = 20
        self.n_action = 10
        self.n_agent = 9
        self.replay_buffer = np.zeros([config.MEMORY_CAPACITY_AC, self.n_state * 2 + 5 + 2 * self.n_action])   # *2代表需要存两个状态，上一时刻与下一时刻，+4是任务、奖励、完成、动作  2*self.n_action实质上是三个后面全局数据收集需要的东西 \probs\one_hot log_probs是一个参数
        self.index = 0
        self.task_cloud = 0
        self.workload = list()
        # self.cloud_network = global_network      # A3C
        # self.cloud_a_optimizer = global_a_optimizer
        # self.cloud_c_optimizer = global_c_optimizer
        # self.actor_critic = global_network # model.ActorCritic(self.n_state, self.n_action)
        #
        # self.actor_critic2 = model.ActorCritic(self.n_state, self.n_action)    # AC    与上面actor_critic有什么区别？
        # self.cloud_network2 = model.ActorCritic(self.n_state, self.n_action)
        # self.cloud_c_optimizer2 = torch.optim.Adam(self.cloud_network2.critic.parameters(), lr=0.01)  #原始0.01
        # self.cloud_a_optimizer2 = torch.optim.Adam(self.cloud_network2.actor.parameters(), lr=0.01)
        self.device = utils.default_device()
        self.dictionary = dictionary
        self.buffer = buffer
        # self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
        self.test_num = dictionary["test_num"]
        # self.env_name = dictionary["env"]
        self.value_lr = dictionary["value_lr"]
        self.policy_lr = dictionary["policy_lr"]
        self.gamma = dictionary["gamma"]
        self.entropy_pen = dictionary["entropy_pen"]
        self.gae_lambda = dictionary["gae_lambda"]
        self.gif = dictionary["gif"]
        # TD lambda
        self.lambda_ = dictionary["lambda"]
        self.experiment_type = dictionary["experiment_type"]
        # Used for masking advantages above a threshold
        self.select_above_threshold = dictionary["select_above_threshold"]

        self.policy_clip = dictionary["policy_clip"]
        self.value_clip = dictionary["value_clip"]
        self.n_epochs = dictionary["n_epochs"]

        self.grad_clip_critic = dictionary["grad_clip_critic"]
        self.grad_clip_actor = dictionary["grad_clip_actor"]

        self.attention_type = dictionary["attention_type"]
        self.avg_agent_group = []
        self.update_ppo_agent = dictionary["update_ppo_agent"]
        self.num_agents = 9
        self.num_actions = self.num_agents + 1
        self.state_num = self.n_state
        # self.compare_network = model.Policy(obs_input_dim=9, num_agents=4,num_actions=5, device='cpu').to(self.device)
        self.num_teams = self.num_agents // 3  # team size is 4
        self.relevant_set = torch.zeros(1, 9, 9).to(self.device)
        team_counter = 0
        for i in range(self.num_agents):
            if i < 3:  # 第一个团队
                self.relevant_set[0][i][:3] = torch.ones(3)  # 第一个团队的成员
            elif i < 6:  # 第二个团队
                self.relevant_set[0][i][3:6] = torch.ones(3)  # 第二个团队的成员
            else:  # 第三个团队
                self.relevant_set[0][i][6:9] = torch.ones(3)  # 第三个团队的成员
        self.non_relevant_set = torch.ones(1, 9, 9).to(self.device) - self.relevant_set

        # print("EXPERIMENT TYPE", self.experiment_type)
        obs_input_dim = self.state_num  # 这部分可能需要改
        self.critic_network = model.Q_network(obs_input_dim=obs_input_dim, num_agents=self.num_agents,
                                              num_actions=self.num_actions, attention_type=self.attention_type,
                                              device=self.device).to(self.device)
        self.critic_network_old = model.Q_network(obs_input_dim=obs_input_dim,
                                                  num_agents=self.num_agents,
                                                  num_actions=self.num_actions, attention_type=self.attention_type,
                                                  device=self.device).to(self.device)

        for param in self.critic_network_old.parameters():
            param.requires_grad_(False)
        # COPY
        self.critic_network_old.load_state_dict(self.critic_network.state_dict())

        self.seeds = [42, 142, 242, 342, 442]
        torch.manual_seed(self.seeds[dictionary["iteration"] - 1])
        # POLICY
        self.policy_network = model.Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents,
                                           num_actions=self.num_actions, device=self.device).to(self.device)
        self.policy_network_old = model.Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents,
                                               num_actions=self.num_actions, device=self.device).to(self.device)
        for param in self.policy_network_old.parameters():
            param.requires_grad_(False)
        # COPY
        self.policy_network_old.load_state_dict(self.policy_network.state_dict())

        # critic_dir = self.dictionary["critic_dir"]
        # try:
        #     os.makedirs(critic_dir, exist_ok=True)
        #     print("Critic Directory created successfully")
        # except OSError as error:
        #     print("Critic Directory can not be created")
        # actor_dir = self.dictionary["actor_dir"]
        # try:
        #     os.makedirs(actor_dir, exist_ok=True)
        #     print("Actor Directory created successfully")
        # except OSError as error:
        #     print("Actor Directory can not be created")
        #
        # if dictionary["load_models"]:
        #     print("LOADING MODELS")
        #     # Loading models
        #     self.critic_network_old.load_state_dict(torch.load(dictionary["critic_dir"]))
        #     self.critic_network.load_state_dict(torch.load(dictionary["critic_dir"]))
        #     self.policy_network_old.load_state_dict(torch.load(dictionary["actor_dir"]))
        #     self.policy_network.load_state_dict(torch.load(dictionary["actor_dir"]))

        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)
        self.cloud = cloud
        random.seed(1)

    def run(self, t):
        t_workload = self.statistic_workload()
        if t_workload > 0:
            self.cloud.submit_workload(self.id, self.statistic_workload())
        self.compute_engine(t)
        self.send_to_cloud(t)
        if t >= 1000 and t % config.MEMORY_CAPACITY_AC == 0 and self.mode == "PRDMAPPO":
        # if self.index > config.MEMORY_CAPACITY_AC and not (t % self.update_ppo_agent) and self.mode == 'PRDMAPPO':
            self.update(t)    # 把全局的网络给复制下来
            #     self.compare_network.load_state_dict(self.policy_network.state_dict())
            # self.replay_buffer = np.zeros([config.MEMORY_CAPACITY_AC, self.n_state * 2 + 5 + 2 * self.n_action])     # 把onehot dist等都存进来
    # 添加更新回合奖励的方法
    def update_episode_rewards(self, reward_value):
        """更新回合奖励并处理回合统计"""
        # 如果是1000的惩罚值，则设置一个上限
        capped_reward = max(-300, -reward_value) if reward_value != 1000 else -300
        
        # 累积到当前回合奖励
        self.current_episode_reward += capped_reward
        self.episode_task_count += 1
        
        # 检查是否达到回合结束条件
        if self.episode_task_count >= self.episode_length:
            # 计算当前回合的平均奖励
            if self.episode_task_count > 0:
                avg_episode_reward = self.current_episode_reward / self.episode_task_count
                self.episode_rewards.append(avg_episode_reward)
                
                # 输出当前回合信息
                # print(f"服务器 {self.id} 完成回合 {self.episode_number}, 平均奖励: {avg_episode_reward:.2f}")
            
            # 重置回合统计
            self.current_episode_reward = 0
            self.episode_task_count = 0
            self.episode_number += 1
    
    def statistic_workload(self):
        total_workload = 0
        for i in range(len(self.compute_queue)):
            total_workload += self.compute_queue[i][0].required_res
        return total_workload

    def get_current_workloads(self):
        current_loads = []
        for i in range(len(self.cloud.workload_log)):
            if len(self.cloud.workload_log[i]) > 0:
                # 获取当前智能体的最新负载（最后一个元素）
                current_loads.append(self.cloud.workload_log[i][-1])
            else:
                # 如果该智能体还没有负载记录，默认负载为0
                current_loads.append(0)
        return current_loads

    def compute_load_variance_penalty(self, res):

        loads = self.get_current_workloads()

        loads_array = np.array(loads)

        tmp = np.zeros_like(loads_array)  
        tmp[self.id] = res  

        # 计算预加载负载
        pre_loads = loads_array - tmp  # 直接减去 tmp
        # print("loads:",loads, "pre_loads:", pre_loads)
        # epslion = 1e-8
        # print(loads)
        # 计算负载的平均值
        mean_load = np.mean(loads)
        mean_preload = np.mean(pre_loads)
        # 计算负载的标准差
        std_loads_dev = np.std(loads)
        std_pre_loads = np.std(pre_loads)
        # 使用标准差作为负载均衡的惩罚项
        # 标准差越大，说明负载分布越不均匀，惩罚越大
        # penalty = std_dev / mean_load  # 归一化惩罚，防止负载值较大时惩罚过大
        penalty = std_loads_dev/mean_load - std_pre_loads/mean_preload
        # print(penalty)
        return penalty

    def calculate_distance_based_bandwidth(self, source_pos, target_pos, base_bandwidth):
        # 计算距离
        distance = math.sqrt((source_pos[0] - target_pos[0])**2 + (source_pos[1] - target_pos[1])**2)
        
        if distance > 1000:
            print(f"警告: 计算出的距离过大: {distance}")
            distance = min(distance, 1000)

        max_distance = 1000.0  # 最大参考距离
        
        attenuation_factor = 1.0 - (distance / max_distance) ** 1.5  # 使用1.5次方使衰减更平缓
        
        attenuation_factor = max(attenuation_factor, 0.0)
        
        min_attenuation = 0.7
        attenuation_factor = max(attenuation_factor, min_attenuation)
        
        effective_bandwidth = base_bandwidth * attenuation_factor
        
        min_bandwidth = base_bandwidth * 0.3
        final_bandwidth = max(effective_bandwidth, min_bandwidth)
        
        # print(f"距离: {distance:.1f}, 衰减因子: {attenuation_factor:.4f}, 有效带宽: {final_bandwidth:.2f}")
        
        return final_bandwidth

    def scheduler(self, t, task, mfn_name):
        if self.mode == 'PRDMAPPO' and self.global_workload:
            workload = self.cloud.pre_workload(t)
        else:
            workload = [0 for _ in range(self.n_agent)]
        vehicle = self.dictionary["all_cars"].get(mfn_name)
        if self.predict :
            if t < 3600 * 12:
                predicted_pos = vehicle.predicted_edge[t].flatten()  # 展平为一维数组
            else:
                predicted_pos = vehicle.predicted_edge[3600 * 12 - 1].flatten()
        else:
            predicted_pos = np.zeros(6,).flatten() 
        
        
        risk = (t - task.start_time)/task.deadline
        task.risk = risk    
        state = [risk, task.result_size, task.task_size, task.required_res, self.f, workload, predicted_pos]   ## 维度变成13了
        action = self.offload(np.hstack(state), t)
        if self.index != 0:
            self.replay_buffer[(self.index - 1) % config.MEMORY_CAPACITY_AC][3+self.n_state:3+self.n_state*2] = np.hstack(state)   # 倒数第self.n_state+1 列开始，一直到倒数第2列结束
        tmp = task.name[task.name.find('_')+1:]
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][0] = float(tmp[tmp.find('_')+1:])      # 任务名称
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][1:self.n_state+1] = np.hstack(state)     # 状态信息
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][self.n_state+1] = action               # 以及动作
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][-1] = 0
        self.index += 1
        if action == 9:  #len(self.sfns):    # 表示任务被发送到了云端, 0-8分别代表9个边缘服务器。
            self.task_cloud += 1
            send_time = task.task_size / config.u2c_band 
            self.send_cloud_queue.append([task, send_time])
        else:
            send_to_sfn = self.sfns[action]    # 发给第几个边缘服务器
            send_to_sfn.compute_queue.append([task, self.id, mfn_name, task.required_res / send_to_sfn.f])

    def compute_engine(self, t):
        total_time = 1
        total_res = 0
        while total_time > 0:
            if len(self.compute_queue) > 0:
                if self.compute_queue[0][-1] <= 1:   
                    total_time -= self.compute_queue[0][-1]
                    total_res = self.compute_queue[0][-1] * self.f   # 计算任务所消耗的资源量，并将其添加到总资源消耗中。
                    t_task = self.compute_queue.pop(0)
                    # 找出车辆当前所在的UAV
                    current_uav_id = -1
                    for uav_id, uav in enumerate(self.sfns):
                        if t_task[2] in uav.bind_car:
                            current_uav_id = uav_id
                            break
                    
                    if t_task[2] in self.bind_car:    # 发送到该服务器
                        
                        send_time = t_task[0].result_size / config.g2u_band
                        completion_time = t  + (1 - total_time) + send_time

                        will_vehicle_stay = self.check_vehicle_in_range(t_task[2], completion_time)
                        
                        if will_vehicle_stay:
                            # 传输成功
                            reward = (t - t_task[0].start_time) + (1 - total_time) + send_time
                        else:
                            # 传输失败，车辆已离开覆盖范围
                            reward = 100  # 大惩罚

                    elif current_uav_id != -1:  # 车辆在另一个UAV的覆盖范围内
                        # 初始化参数
                        total_uav_transfer_time = 0  # 总UAV间传输时间
                        vehicle_time = t_task[0].result_size / config.g2u_band  # UAV到车辆的传输时间
                        
                        # 开始计算传输路径
                        source_uav_id = self.id  # 当前UAV
                        target_uav_id = current_uav_id  # 初始目标UAV (车辆当前所在UAV)
                        current_time = t + (1 - total_time)  # 计算开始的时间点
                        max_hops = 3  # 最大UAV间跳转次数，防止无限循环
                        hops = 0  # 跳转计数器
                        uav_path = [source_uav_id]  # 传输路径，记录经过的UAV
                        transmit_success = False  # 传输是否成功
                        
                        while hops < max_hops:
                            # 计算从当前UAV到目标UAV的传输时间
                            source_pos = self.sfns[source_uav_id].pos
                            target_pos = self.sfns[target_uav_id].pos
                            # print(f"source_pos: {source_pos}, target_pos: {target_pos}")
                            effective_uav_bandwidth = self.calculate_distance_based_bandwidth(
                            source_pos, target_pos, config.core_band
                        )
                            uav_to_uav_time = t_task[0].result_size / effective_uav_bandwidth

                            # distance = math.sqrt((source_pos[0] - target_pos[0])**2 + (source_pos[1] - target_pos[1])**2)
                            # if distance <= 300:
                            #     uav_to_uav_time = t_task[0].result_size / (config.core_band * 0.95)
                            # elif distance <= 600:
                            #     uav_to_uav_time = t_task[0].result_size / (config.core_band * 0.9)
                            # else:
                            #     uav_to_uav_time = t_task[0].result_size / (config.core_band * 0.85)

                            # uav_to_uav_time = t_task[0].result_size / config.core_band
                            
                            total_uav_transfer_time += uav_to_uav_time
                            
                            # 更新当前时间，包括UAV间传输时间
                            current_time += uav_to_uav_time
                            uav_path.append(target_uav_id)  # 添加目标UAV到路径
                            
                            # 计算在目标UAV完成传输到车辆的时间点
                            completion_time = current_time + vehicle_time
                            
                            # 预测车辆在完成时刻是否仍在目标UAV覆盖范围内
                            target_uav = self.sfns[target_uav_id]
                            will_vehicle_stay = target_uav.check_vehicle_in_range(t_task[2], completion_time)
                            
                            if will_vehicle_stay:
                                # 传输成功
                                transmit_success = True
                                reward = (t - t_task[0].start_time) + (1 - total_time) + total_uav_transfer_time + vehicle_time
                                break
                            else:
                                # 车辆将离开当前目标UAV，查找车辆在完成时刻的位置
                                next_uav_id = self.find_vehicle_future_position(t_task[2], completion_time)
                                
                                if next_uav_id != -1 and next_uav_id != target_uav_id:
                                    # 车辆移动到另一个UAV，更新目标并继续
                                    # source_uav_id = target_uav_id
                                    target_uav_id = next_uav_id
                                    hops += 1
                                else:
                                    # 车辆离开所有UAV覆盖范围，传输失败
                                    transmit_success = False
                                    reward = 100  # 大惩罚
                                    break
                        
                        # 如果达到最大跳转次数仍未成功，视为失败
                        if hops >= max_hops:
                            reward = 100  # 大惩罚
                            transmit_success = False

                    else:   # cy分析：发送到云 解决跨域问题
                        reward = (t - t_task[0].start_time) + (1 - total_time) + self.cloud.wait_time('send', t) + \
                                 t_task[0].result_size / config.u2c_band  + t_task[0].result_size / config.g2h_band
                    if reward <= t_task[0].deadline:
                        self.task_success += 1
                        self.task_res += t_task[0].required_res
                        self.time_counter += reward
                    else:
                        self.task_drop += 1
                        reward = 100
                    
                    send_sfn = self.sfns[t_task[1]]
                    tmp = t_task[0].name[t_task[0].name.find('_') + 1:]  # 如果找到了下划线，返回该下划线的索引（位置），从 0 开始计数
                    if reward != 100:

                        load_variance = self.compute_load_variance_penalty(total_res)
                        beta = 3
                        load_penalty = beta * load_variance  # beta是负载均衡惩罚强度
                        reward += load_penalty

                    current_workload = self.get_current_workloads()
                    for k in range(len(send_sfn.replay_buffer)):
                        if send_sfn.replay_buffer[k][0] == float(tmp[tmp.find('_')+1:]):
                            send_sfn.replay_buffer[k][self.n_state+2] = -reward   # 11的位置是奖励   # reward越接近0当然时延越小，越是我们想选的动作，所以应该更大
                            send_sfn.replay_buffer[k][-1] = 1  # 任务完成
                    
                    self.update_episode_rewards(reward)
                else:
                    self.compute_queue[0][-1] -= 1
                    break
            else:
                break

        self.workload.append(total_res)

    def check_vehicle_in_range(self, vehicle_id, future_time):

        vehicle = self.dictionary["all_cars"][vehicle_id]
        
        future_t = int(future_time)
        
        # 检查时间是否超出轨迹长度
        if future_t >= len(vehicle.route):
            return False
        
        # 获取车辆在future_t时刻的位置
        future_pos = vehicle.route[future_t]
        
        # 计算车辆与当前UAV的距离
        distance = math.sqrt((future_pos[0] - self.pos[0])**2 + (future_pos[1] - self.pos[1])**2)
        
        # 检查距离是否在UAV覆盖范围内
        return distance <= config.base_cover

    def find_vehicle_future_position(self, vehicle_id, future_time):
        """
        预测车辆在未来时刻所在的UAV ID
        
        Args:
            vehicle_id (str): 车辆ID
            future_time (float): 未来时刻
        
        Returns:
            int: 车辆所在UAV的ID，如果不在任何UAV的覆盖范围内，则返回-1
        """
        # 获取车辆对象
        vehicle = self.dictionary["all_cars"][vehicle_id]
        
        # 计算未来时间对应的时间步
        future_t = int(future_time)
        
        # 检查时间是否超出轨迹长度
        if future_t >= len(vehicle.route):
            return -1
        
        # 获取车辆在future_t时刻的位置
        future_pos = vehicle.route[future_t]
        
        # 检查该位置在哪个UAV的覆盖范围内
        for uav_id, uav in enumerate(self.sfns):
            distance = math.sqrt((future_pos[0] - uav.pos[0])**2 + (future_pos[1] - uav.pos[1])**2)
            if distance <= config.base_cover:
                return uav_id
        
        return -1  # 不在任何UAV覆盖范围内

    def send_to_cloud(self, t):
        total_time = 5
        while total_time > 0:
            if len(self.send_cloud_queue) > 0:
                if self.send_cloud_queue[0][1] <= total_time:

                    total_time -= self.send_cloud_queue[0][1]
                    t_task = self.send_cloud_queue.pop(0)
                    reward = self.cloud.receive_task(t_task[0], t)   # 补充卸载到云的任务增加奖励设置
                    tmp = t_task[0].name[t_task[0].name.find('_') + 1:]
                    for k in range(len(self.replay_buffer)):
                        if self.replay_buffer[k][0] == float(tmp[tmp.find('_') + 1:]):
                            self.replay_buffer[k][self.n_state + 2] = -reward  # 11的位置是奖励
                            self.replay_buffer[k][-1] = 1  # 任务完成状况
                else:
                    self.send_cloud_queue[0][1] -= total_time
                    total_time = 0
            else:
                break

    def receive_task(self, t, task, mfn_name):
        self.scheduler(t, task, mfn_name)

    def offload(self, state, t):
        if self.mode == 'PRDMAPPO':
            if t >= 1000:
                action = self.get_action(state)
            else:
                action = random.randint(0, self.n_action-1)
        elif self.mode == 'Heuristic':
            risk = state[0]  
            task_size = state[2]  
            required_res = state[3]  
            

            edge_workloads = self.cloud.pre_workload(t)
            
            if len(state) > 20:  # 确保状态向量包含预测数据
                predicted_pos = state[5:11]  # 假设位置6-11是预测数据
                

                server_scores = np.zeros(self.n_action)
                

                max_workload = max(edge_workloads) if max(edge_workloads) > 0 else 1
                load_scores = 1 - (np.array(edge_workloads) / max_workload)
                

                future_coverage = np.zeros(self.n_action)
                valid_predictions = 0
                

                for pos in predicted_pos:
                    if pos >= 0 and pos < 10:  
                        future_coverage[int(pos)] += 1
                        valid_predictions += 1

                if valid_predictions > 0:
                    future_coverage = future_coverage / valid_predictions
                

                urgency_factor = min(1.0, risk)  
                

                workload_weight = 0.4
                trajectory_weight = 0.6
                

                if urgency_factor > 0.7: 
                    trajectory_weight = 0.8
                    workload_weight = 0.2
                

                for i in range(9):  
                    server_scores[i] = workload_weight * load_scores[i] + trajectory_weight * future_coverage[i]
                

                cloud_threshold = 0.2  
                best_edge_score = max(server_scores[:9])
                

                if best_edge_score < cloud_threshold or urgency_factor > 0.9 or required_res > 2 * self.f:

                    if random.random() < 0.1:
                        return 9 
                

                best_server = np.argmax(server_scores[:9])
                return best_server
            else:
                # 简单版本：95%概率选择负载最小的服务器，5%概率选择云服务器
                if random.random() < 0.95:
                    return np.argmin(edge_workloads[:9])  # 选择负载最小的边缘服务器
                else:
                    return 9  # 云服务器
        elif self.mode == 'random':
            action = random.randint(0, self.n_action-1)
        else:
            action = self.id
        return action



    def get_action(self, state_policy, greedy=False):

        with torch.no_grad():
            # 将输入状态转换为 torch 张量，并调整维度
            state_policy = torch.from_numpy(state_policy).float().to(self.device).unsqueeze(0)

            # 获取动作分布
            dist = self.policy_network(state_policy).squeeze(0)

            if greedy:
                # 使用贪心策略选择动作
                action = dist.argmax().detach().cpu().item()
            else:
                # # 取动作分布的倒数并归一化
                # inv_dist = 1.0 / (dist + 1e-6)  # 加1e-6防止除零
                # inv_dist /= inv_dist.sum()  # 归一化

                action = Categorical(dist).sample().detach().cpu().item()

            # 计算动作的 one-hot 编码
            one_hot_action = np.zeros(self.n_action)
            one_hot_action[action] = 1

            probs = Categorical(dist)
            action_logprob = probs.log_prob(torch.tensor(action).to(self.device)).item()
            # print("dist_shape:", dist.shape)
            # print("buffer.shape",self.replay_buffer.shape)
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2 : 3 + self.n_state * 2 + self.n_action] = dist.cpu().numpy()
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2+self.n_action : 3+self.n_state*2+self.n_action*2] = one_hot_action
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2+self.n_action*2] = action_logprob
            # # 保存动作分布、动作的 log 概率和 one-hot 编码
            # self.buffer.probs[self.index, self.id, :] = dist_probs_numpy
            # self.buffer.logprobs[self.index, self.id, :] = action_logprob_numpy
            # self.buffer.one_hot_actions[self.index, self.id, :] = one_hot_action
            return action


    def compare_model_params(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(param1, param2):
                print("Parameters are different.")
        print("Parameters are the same.")

    def calculate_advantages(self, values, rewards, dones):
        advantages = []
        next_value = 0
        advantage = 0
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        masks = 1 - dones
        for t in reversed(range(0, len(rewards))):
            td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
            next_value = values.data[t]

            advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
            advantages.insert(0, advantage)

        advantages = torch.stack(advantages)

        return advantages
    def find_vehicle_uav(self, vehicle_id):
        """查找特定车辆当前所在的UAV"""
        for uav_id, uav in enumerate(self.sfns):
            if vehicle_id in uav.bind_car:
                return uav_id
        return -1  # 如果车辆不在任何UAV覆盖范围内

    def calculate_deltas(self, values, rewards, dones):
        deltas = []
        next_value = 0
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        masks = 1 - dones
        for t in reversed(range(0, len(rewards))):
            td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
            next_value = values.data[t]
            deltas.insert(0, td_error)
        deltas = torch.stack(deltas)

        return deltas

    def nstep_returns(self, values, rewards, dones):
        deltas = self.calculate_deltas(values, rewards, dones)
        advs = self.calculate_returns(deltas, self.gamma * self.lambda_)
        target_Vs = advs + values
        return target_Vs

    def calculate_returns(self, rewards, discount_factor):
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)

        returns_tensor = torch.stack(returns).to(self.device)

        return returns_tensor



    def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
        advantage = None
        masking_advantage = None
        mean_min_weight_value = -1
        if "shared" in self.experiment_type:
            advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones), dim=-2).detach()
        elif "prd_above_threshold" in self.experiment_type:
            masking_advantage = (weights_prd > self.select_above_threshold).int()
            advantage = torch.sum(
                self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage, -1, -2),
                dim=-2)

        return advantage, masking_advantage

    # def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
    #     # 添加调试输出以了解张量形状
    #     # print(f"Debug - V_values shape: {V_values.shape}")
    #     # print(f"Debug - rewards shape: {rewards.shape}")
    #     # print(f"Debug - dones shape: {dones.shape}")
    #     # print(f"Debug - weights_prd shape: {weights_prd.shape}")
        
    #     advantage = None
    #     masking_advantage = None
        
    #     if "shared" in self.experiment_type:
    #         advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones), dim=-2).detach()
    #     elif "prd_above_threshold" in self.experiment_type:
    #         # 创建掩码
    #         masking_advantage = (weights_prd > self.select_above_threshold).int()
            
    #         # 计算advantages
    #         advantages = self.calculate_advantages(V_values, rewards, dones)
    #         # print(f"Debug - advantages shape: {advantages.shape}")
    #         # print(f"Debug - masking_advantage shape: {masking_advantage.shape}")
            
    #         # 确保维度匹配
    #         if masking_advantage.shape[-1] != advantages.shape[-2]:
    #             # print(f"维度不匹配! masking_advantage: {masking_advantage.shape[-1]}, advantages: {advantages.shape[-2]}")
                
    #             # 调整掩码维度
    #             if masking_advantage.shape[-1] > advantages.shape[-2]:
    #                 # 如果掩码维度大于优势维度，则裁剪掩码
    #                 # print(f"裁剪掩码从 {masking_advantage.shape[-1]} 到 {advantages.shape[-2]}")
    #                 masking_advantage = masking_advantage[..., :advantages.shape[-2]]
    #             else:
    #                 # 如果掩码维度小于优势维度，则填充掩码
    #                 padding = torch.zeros(*masking_advantage.shape[:-1], advantages.shape[-2] - masking_advantage.shape[-1], 
    #                                     device=masking_advantage.device)
    #                 masking_advantage = torch.cat([masking_advantage, padding], dim=-1)
    #                 # print(f"填充掩码从 {masking_advantage.shape[-1] - padding.shape[-1]} 到 {masking_advantage.shape[-1]}")
            
    #         # print(f"调整后 masking_advantage shape: {masking_advantage.shape}")
            
    #         # 执行乘法操作
    #         advantages_masked = advantages * torch.transpose(masking_advantage, -1, -2)
    #         advantage = torch.sum(advantages_masked, dim=-2)
        
    #     return advantage, masking_advantage

    

    def update(self, episode):
        # convert list to tensor
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device).detach()
        old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device).detach()
        old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device).detach()
        old_probs = torch.FloatTensor(np.array(self.buffer.probs)).to(self.device).detach()
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs)).to(self.device).detach()
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device).detach()
        dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device).detach()

        # print("actions:", old_actions.shape)  # 这里是0
        # print(old_actions)
        # print("states:", old_states.shape)
        # print(old_states)
        # print("probs:", old_probs.shape)
        # print(old_probs)
        # print("one_hot_actions:", old_one_hot_actions.shape)
        # print(old_one_hot_actions)
        # print("logprobs:", old_logprobs.shape)
        # print(old_logprobs)
        # print("reward:", rewards.shape)
        # print(rewards)
        # print("dones:", dones.shape)
        # print(dones)

        Values_old, Q_values_old, weights_value_old = self.critic_network_old(old_states, old_probs.squeeze(-2),
                                                                              old_one_hot_actions)
        Values_old = Values_old.reshape(-1, self.num_agents, self.num_agents)

        Q_value_target = self.nstep_returns(Q_values_old, rewards, dones).detach()

        value_loss_batch = 0
        policy_loss_batch = 0
        entropy_batch = 0
        value_weights_batch = None
        grad_norm_value_batch = 0
        grad_norm_policy_batch = 0
        agent_groups_over_episode_batch = 0
        avg_agent_group_over_episode_batch = 0

        # torch.autograd.set_detect_anomaly(True)
        # Optimize policy for n epochs
        for _ in range(self.n_epochs):

            Value, Q_value, weights_value = self.critic_network(old_states, old_probs.squeeze(-2),
                                                                old_one_hot_actions)
            Value = Value.reshape(-1, self.num_agents, self.num_agents)

            advantage, masking_advantage = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value,
                                                                                  episode)

            if "threshold" in self.experiment_type:
                agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2), dim=0) / \
                                            masking_advantage.shape[0]
                avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
                agent_groups_over_episode_batch += agent_groups_over_episode
                avg_agent_group_over_episode_batch += avg_agent_group_over_episode

            dists = self.policy_network(old_states)
            probs = Categorical(dists.squeeze(0))
            logprobs = probs.log_prob(old_actions)

            critic_loss_1 = F.smooth_l1_loss(Q_value, Q_value_target)
            critic_loss_2 = F.smooth_l1_loss(
                torch.clamp(Q_value, Q_values_old - self.value_clip, Q_values_old + self.value_clip), Q_value_target)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            # Finding Surrogate Loss
            surr1 = ratios * advantage.detach()
            surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantage.detach()

            # final loss of clipped objective PPO
            entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=2))
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen * entropy

            entropy_weights = -torch.mean(
                torch.sum(weights_value * torch.log(torch.clamp(weights_value, 1e-10, 1.0)), dim=2))
            critic_loss = torch.max(critic_loss_1, critic_loss_2)

            # take gradient step
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.grad_clip_critic)
            self.critic_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
            self.policy_optimizer.step()

            if self.dictionary["load_models"]:
                print("LOADING MODELS")
                # Loading models
                self.critic_network_old.load_state_dict(torch.load(self.dictionary["critic_dir"] + "critic_network.pt"))
                self.critic_network.load_state_dict(torch.load(self.dictionary["critic_dir"] + "critic_network.pt"))
                self.policy_network_old.load_state_dict(torch.load(self.dictionary["actor_dir"] + "actor_network.pt"))
                self.policy_network.load_state_dict(torch.load(self.dictionary["actor_dir"] + "actor_network.pt"))

            value_loss_batch += critic_loss
            policy_loss_batch += policy_loss
            entropy_batch += entropy
            grad_norm_value_batch += grad_norm_value
            grad_norm_policy_batch += grad_norm_policy
            if value_weights_batch is None:
            #     value_weights_batch = torch.zeros_like(weights_value.cpu())
            # value_weights_batch += weights_value.detach().cpu()
                value_weights_batch = torch.zeros_like(weights_value)
            value_weights_batch += weights_value.detach()

        # Copy new weights into old policy
        self.policy_network_old.load_state_dict(self.policy_network.state_dict())

        # Copy new weights into old critic
        self.critic_network_old.load_state_dict(self.critic_network.state_dict())


        torch.save(self.critic_network.state_dict(), self.dictionary["critic_dir"] + "critic_network.pt")
        torch.save(self.policy_network.state_dict(), self.dictionary["actor_dir"] + "actor_network.pt")

        # clear buffer
        # self.buffer.clear()

        value_loss_batch /= self.n_epochs
        policy_loss_batch /= self.n_epochs
        entropy_batch /= self.n_epochs
        grad_norm_value_batch /= self.n_epochs
        grad_norm_policy_batch /= self.n_epochs
        value_weights_batch /= self.n_epochs
        agent_groups_over_episode_batch /= self.n_epochs
        avg_agent_group_over_episode_batch /= self.n_epochs

        if "prd" in self.experiment_type:
            num_relevant_agents_in_relevant_set = self.relevant_set * masking_advantage
            num_non_relevant_agents_in_relevant_set = self.non_relevant_set * masking_advantage
            true_negatives = self.non_relevant_set * (1 - masking_advantage)
        else:
            num_relevant_agents_in_relevant_set = None
            num_non_relevant_agents_in_relevant_set = None
            true_negatives = None



        threshold = self.select_above_threshold
        self.plotting_dict = {
            "value_loss": value_loss_batch,
            "policy_loss": policy_loss_batch,
            "entropy": entropy_batch,
            "grad_norm_value": grad_norm_value_batch,
            "grad_norm_policy": grad_norm_policy_batch,
            "weights_value": value_weights_batch,
            "num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
            "num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
            "true_negatives": true_negatives,
            "threshold": threshold
        }

        if "threshold" in self.experiment_type:
            self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
            self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch