import math
import random

import torch
from torch import optim
from tqdm import tqdm
import logging
from HAP import HAP
from utils import Task
import numpy as np
import config
import model
from SFN import SFN
from utils import default_device
from torch.distributions import Categorical
import torch.nn.functional as F

class Ground:   # Mobile Fog Node 移动雾节点
    def __init__(self, id, f, c,  name, sfns, cloud, mode, task_gen_rate, dictionary, buffer, agent_num):
        self.predict = dictionary["prediction"]
        self.dictionary = dictionary
        self.id = id
        self.f = f   # 设备计算能力
        self.c = c   # 系统的内存容量
        self.name = name
        self.task_gen_rate = task_gen_rate
        # 任务执行情况统计
        self.task_success = 0
        self.task_drop = 0
        self.send_sfn_drop = 0
        self.send_sfn_success = 0
        self.time_counter = 0
        self.send_cloud_success = 0
        self.send_cloud_drop = 0
        self.energy_comsumpution = 0
        self.success_energy_num = 0
        self.x = 0
        self.y = 0
        # 添加回合奖励跟踪
        self.episode_rewards = []          # 存储每个回合的平均奖励
        self.current_episode_reward = 0    # 当前回合累积奖励
        self.episode_task_count = 0        # 当前回合处理的任务数
        self.episode_length = 50           # 每回合包含的任务数量
        self.episode_number = 0            # 当前回合编号
        
        # 添加任务追踪字典，用于关联任务ID和奖励
        self.task_rewards = {}   

        # 区域内的边缘服务器和云
        self.sfns = sfns
        self.cloud = cloud
        self.bind_sfn = None
        self.mode = mode
        self.send_cloud_queue = list()
        self.send_sfn_queue = list()
        self.compute_queue = list()
        # 加载路径和任务
        self.route = np.load('./route/' + self.name + '_route.npy', allow_pickle=True)
        self.task = self.load_task('./task/' + self.name + '_task.npy') # size/4-8
        self.speed = np.load(f'./route/' + self.name + '_speeds.npy', allow_pickle=True)
        self.predicted_edge = np.load(f'./predict_result/edge_coverage/'+ self.name + '_edge_coverage.npy', allow_pickle=True) 
        self.N_STATE = 12
        self.N_ACTIONS = 3
        self.N_Agent = agent_num
        # 定义重放内存
        self.replay_buffer = np.zeros((config.MEMORY_CAPACITY, self.N_STATE * 2 + 4 + 2 * self.N_ACTIONS + 1))  # 增加一个done值
        self.index = 0
        self.offload_point = list()
        self.mode = mode
        self.end_time = 0
        self.task_exec_count = 0
        self.device = default_device()
        self.car_num = self.N_Agent
        self.car_state = self.N_STATE
        self.car_actions = self.N_ACTIONS
        self.car_buffer = buffer
        self.test_num = dictionary["test_num"]
        # self.env_name = dictionary["env"]
        self.value_lr = dictionary["car_value_lr"]
        self.policy_lr = dictionary["car_policy_lr"]
        self.gamma = dictionary["car_gamma"]
        self.entropy_pen = dictionary["car_entropy_pen"]
        self.gae_lambda = dictionary["car_gae_lambda"]
        self.gif = dictionary["gif"]
        # TD lambda
        self.lambda_ = dictionary["car_lambda"]
        self.experiment_type = dictionary["experiment_type"]
        # Used for masking advantages above a threshold
        self.select_above_threshold = dictionary["select_above_threshold"]

        self.policy_clip = dictionary["car_policy_clip"]
        self.value_clip = dictionary["car_value_clip"]
        self.n_epochs = dictionary["car_n_epochs"]
        self.log_count = 0
        self.grad_clip_critic = dictionary["car_grad_clip_critic"]
        self.grad_clip_actor = dictionary["car_grad_clip_actor"]

        self.attention_type = dictionary["car_attention_type"]
        self.avg_agent_group = []
        self.update_ppo_agent = dictionary["car_update"]
        self.car_teams = self.car_num // 3  # team size is 4
        self.car_relevant_set = torch.zeros(1, self.N_Agent, self.N_Agent).to(self.device)
        car_team_counter = 0
        car_team_counter = 0
        for i in range(self.car_num):
            if i < 4:
                self.car_relevant_set[0][i][:4] = torch.ones(4)
            elif i >= 4 and i < 8:
                self.car_relevant_set[0][i][4:8] = torch.ones(4)
            elif i >= 8 and i < 12:
                self.car_relevant_set[0][i][8:12] = torch.ones(4)
            
        self.car_non_relevant_set = torch.ones(1, self.N_Agent, self.N_Agent).to(self.device) - self.car_relevant_set

        car_obs_input_dim = self.car_state  # 这部分可能需要改
        self.car_critic_network = model.Q_network(obs_input_dim=car_obs_input_dim, num_agents=self.car_num,
                                                  num_actions=self.car_actions, attention_type=self.attention_type,
                                                  device=self.device).to(self.device)
        self.car_critic_network_old = model.Q_network(obs_input_dim=car_obs_input_dim,
                                                      num_agents=self.car_num,
                                                      num_actions=self.car_actions, attention_type=self.attention_type,
                                                      device=self.device).to(self.device)
        for param in self.car_critic_network_old.parameters():
            param.requires_grad_(False)
        # COPY
        self.car_critic_network_old.load_state_dict(self.car_critic_network.state_dict())

        self.seeds = [42, 142, 242, 342, 442]
        torch.manual_seed(self.seeds[dictionary["iteration"] - 1])
        # POLICY
        self.car_policy_network = model.Policy(obs_input_dim=car_obs_input_dim, num_agents=self.car_num,
                                               num_actions=self.car_actions, device=self.device).to(self.device)
        self.car_policy_network_old = model.Policy(obs_input_dim=car_obs_input_dim, num_agents=self.car_num,
                                                   num_actions=self.car_actions, device=self.device).to(self.device)
        for param in self.car_policy_network_old.parameters():
            param.requires_grad_(False)

        self.car_critic_optimizer = optim.Adam(
            self.car_critic_network.parameters(), 
            lr=float(dictionary["car_value_lr"]),
            weight_decay=1e-4
        )

        self.car_policy_optimizer = optim.AdamW(
            self.car_policy_network.parameters(), 
            lr=float(dictionary["car_policy_lr"]), 
            weight_decay=1e-4
        )
        # COPY
        self.car_policy_network_old.load_state_dict(self.car_policy_network.state_dict())

    # 添加一个新方法来处理任务奖励的累积和回合划分
    def update_episode_rewards(self, task_id, reward_value):
        """更新任务奖励并处理回合统计"""
        # 如果是惩罚值1000，则设置一个上限
        capped_reward = max(-30, -reward_value) if reward_value != 100 else -30
        
        # 累加到当前回合奖励
        self.current_episode_reward += capped_reward
        self.episode_task_count += 1
        
        # 检查是否达到回合结束条件
        if self.episode_task_count >= self.episode_length:
            # 计算当前回合的平均奖励
            if self.episode_task_count > 0:
                avg_episode_reward = self.current_episode_reward / self.episode_task_count
                self.episode_rewards.append(avg_episode_reward)
                
                # 输出当前回合信息
                # print(f"车辆 {self.id} 完成回合 {self.episode_number}, 平均奖励: {avg_episode_reward:.2f}")
            
            # 重置回合统计
            self.current_episode_reward = 0
            self.episode_task_count = 0
            self.episode_number += 1
            
            # 清空任务奖励字典，避免内存占用过大
            self.task_rewards.clear()
    # 相应的，在读取任务时需要将字典格式转换回Task对象
    def load_task(self, file_path):
        tasks = np.load(file_path, allow_pickle=True)
        converted_tasks = []
        task_index = 0  # 用于保持连续的任务索引
        
        for time_step_tasks in tasks:
            current_tasks = []
            for task in time_step_tasks:
                # 从任务名称中提取设备名称
                device_name = task.name[:len(self.name)]  # 获取 'car' 部分
                
                # 创建新任务时使用连续的索引
                new_task = Task(device_name, task_index, task.start_time)
                
                # 设置任务的其他属性
                new_task.task_size = task.task_size
                new_task.result_size = task.result_size
                new_task.deadline = task.deadline
                new_task.required_res = task.required_res
                new_task.end_time = task.end_time
                
                current_tasks.append(new_task)
                task_index += 1  # 增加任务索引
                
            converted_tasks.append(current_tasks)
        
        return converted_tasks

    def run(self, t):
        if t < 3600 * 12:   # 为什么后面2000个时间步不扫描和分配任务了
        # if t < 3600 * 5:
            self.scan_uav(t)
            self.scheduler(t)
        self.send_to_uav(t)
        self.send_to_hap(t)
        self.compute_engine(t)
        if self.index > config.MEMORY_CAPACITY and not(t % self.update_ppo_agent) and self.mode=='PRDMAPPO':
            self.car_update(t)

    def scheduler(self, t):  # 遍历t时间步下的所有任务
        if(self.N_ACTIONS == 3):
            for i in range(len(self.task[t])):
                if random.randint(1, 10) > self.task_gen_rate * 10:     # 任务生成率0.4  在模拟任务调度系统时，需要考虑任务生成的随机性
                    continue
                self.task_exec_count += 1
                t_task = self.task[t][i]   # t时间步下的第i个任务
                
                self.end_time += t_task.deadline

                if self.bind_sfn is None :
                    if not(self.wait_time(t, 'local') + t_task.required_res / self.f <= t_task.deadline and self.memory_check(t_task)):
                        wait_cloud_time = self.wait_time(t, 'cloud')
                        send_time = t_task.task_size / config.g2h_band  
                        self.send_cloud_queue.append([t_task, send_time, wait_cloud_time + t + send_time])
                    else:    
                        self.compute_queue.append([t_task, t_task.required_res / self.f,
                                                self.wait_time(t, 'local') + t + t_task.required_res / self.f])   # 计算队列存储的是 1.任务 2.任务计算时间 3.任务完成时间
                        energy_reward =  config.kappa * (self.f ** 2) * t_task.required_res
                        self.energy_comsumpution += energy_reward
                        self.success_energy_num += 1
                else:
                    
                    if self.predict:
                        predicted_edge = self.predicted_edge[t].flatten()  # 展平为一维数组
                    else:
                        predicted_edge = np.zeros(6,).flatten() 
                        
                    state = np.concatenate([[len(self.bind_sfn.compute_queue), self.f, 
                                    self.bind_sfn.f, t_task.task_size, t_task.required_res, config.g2u_band], predicted_edge])
                    action = self.offload(t, t_task, state)
                    # if action == 1:
                    #     print(f"卸载到边缘{self.x}")
                    #     self.x += 1
                    reward = 0

                    # 为任务创建唯一ID (使用任务名称的一部分作为ID)
                    task_id = float(t_task.name[len(self.name) + 1:])
                    
                    # 初始化任务奖励记录
                    self.task_rewards[task_id] = 0

                    if action == 0:  # 代表卸载到本地
                        if self.wait_time(t, 'local') + t_task.required_res / self.f <= t_task.deadline and self.memory_check(t_task):
                            self.compute_queue.append([t_task, t_task.required_res / self.f,
                                                    self.wait_time(t, 'local') + t + t_task.required_res / self.f])   # 计算队列存储的是 1.任务 2.任务计算时间 3.任务完成时间
                            delay_reward = self.wait_time(t, 'local') + t_task.required_res / self.f
                            energy_reward =  config.kappa * (self.f ** 2) * t_task.required_res
                            self.energy_comsumpution += energy_reward
                            self.success_energy_num += 1
                            # reward = config.r_w * delay_reward + (1 - config.r_w) * energy_reward
                            reward = delay_reward
                            # print(f"local: delay_reward:{delay_reward} energy_reward:{energy_reward} reward:{reward}")
                        else:
                            self.task_drop += 1
                            reward = 100
                        # 更新任务奖励
                        self.task_rewards[task_id] = reward
                        
                        # 直接在本地处理的任务即可更新回合奖励
                        self.update_episode_rewards(task_id, reward)
                    elif action == 1:   # 1代表卸载到边缘服务器
                        wait_sfn_time = self.wait_time(t, 'sfn')
                        # print("wait_sfn_time:", wait_sfn_time)
                        send_time = t_task.task_size / config.g2u_band   # 边缘服务器和车辆通过V2I进行通信。
                        self.send_sfn_queue.append([t_task, send_time, wait_sfn_time + t + send_time])
                    elif action == 2:  # 2 代表卸载到云服务器
                        wait_cloud_time = self.wait_time(t, 'cloud')
                        send_time = t_task.task_size / config.g2h_band
                        self.send_cloud_queue.append([t_task, send_time, wait_cloud_time + t + send_time])
                        # delay_reward = (wait_cloud_time / 5)  + send_time     # wait_cloud_time / 5 , 5 是模拟并行真实的等待时间。
                        delay_reward = wait_cloud_time  + send_time
                        energy_reward = config.p_tx * send_time * 2
                        self.energy_comsumpution += energy_reward
                        self.success_energy_num += 1
                        # reward = config.r_w * delay_reward + (1-config.r_w) * energy_reward
                        reward = delay_reward
                        self.task_rewards[task_id] = reward
                        
                        # 直接在本地处理的任务即可更新回合奖励
                        self.update_episode_rewards(task_id, reward)
                    self.replay_buffer[(self.index - 1) % config.MEMORY_CAPACITY][self.N_STATE + 2] = -reward
        else:
            for i in range(len(self.task[t])):
                if random.randint(1, 10) > self.task_gen_rate * 10:     # 任务生成率0.4  在模拟任务调度系统时，需要考虑任务生成的随机性
                    continue
                self.task_exec_count += 1
                t_task = self.task[t][i]   # t时间步下的第i个任务
                task_id = float(t_task.name[len(self.name) + 1:])
                self.task_rewards[task_id] = 0

                self.end_time += t_task.deadline
                if self.wait_time(t, 'local') + t_task.required_res / self.f <= t_task.deadline and self.memory_check(t_task):  # 本地计算
                    self.compute_queue.append([t_task, t_task.required_res / self.f,
                                            self.wait_time(t, 'local') + t + t_task.required_res / self.f])   # 计算队列存储的是 1.任务 2.任务计算时间 3.任务完成时间
                    energy_reward =  config.kappa * (self.f ** 2) * t_task.required_res
                    self.energy_comsumpution += energy_reward
                    self.success_energy_num += 1
                else:
                    if self.bind_sfn is None:  # 如果没有绑定SFN，就发送到云端
                        wait_cloud_time = self.wait_time(t, 'cloud')
                        send_time = t_task.task_size / config.g2h_band  
                        self.send_cloud_queue.append([t_task, send_time, wait_cloud_time + t + send_time])
                    else:
                        if self.predict:
                            predicted_edge = self.predicted_edge[t].flatten()  # 展平为一维数组
                        else:
                            predicted_edge = np.zeros(6,).flatten() 
                        
                        state = np.concatenate([predicted_edge, [len(self.bind_sfn.compute_queue), self.f, 
                                    self.bind_sfn.f, t_task.task_size, t_task.required_res, config.g2u_band]])

                        action = self.offload(t, t_task, state)
                        if action == 0:   # 0代表卸载到边缘服务器
                            wait_sfn_time = self.wait_time(t, 'sfn')
                            # print("wait_sfn_time:", wait_sfn_time)
                            send_time = t_task.task_size / config.g2u_band   # 边缘服务器和车辆通过V2I进行通信。
                            self.send_sfn_queue.append([t_task, send_time, wait_sfn_time + t + send_time])
                        elif action == 1:  # 1 代表卸载到云服务器
                            wait_cloud_time = self.wait_time(t, 'cloud')                        
                            send_time = t_task.task_size / config.g2h_band
                            self.send_cloud_queue.append([t_task, send_time, wait_cloud_time + t + send_time])
                            # delay_reward = (wait_cloud_time / 5)  + send_time     # wait_cloud_time / 5 , 5 是模拟并行真实的等待时间。
                            delay_reward = wait_cloud_time + send_time
                            energy_reward = config.p_tx * send_time * 2
                            self.energy_comsumpution += energy_reward
                            self.success_energy_num += 1
                            # reward = config.r_w * delay_reward + (1-config.r_w) * energy_reward
                            reward = delay_reward  
                            self.task_rewards[task_id] = reward
                    
                            # 直接在本地处理的任务即可更新回合奖励
                            self.update_episode_rewards(task_id, reward)

                            self.replay_buffer[(self.index - 1) % config.MEMORY_CAPACITY][self.N_STATE + 2] = -reward

    def compute_engine(self, t):  # 模拟了计算引擎按照时间片执行任务的过程。
        total_time = 1   # 表示每个任务需要的执行时间，表示每个时间片
        while total_time > 0:
            if len(self.compute_queue) > 0:
                if self.compute_queue[0][1] <= 1:    # 说明该任务可以在这个时间片内完成
                    total_time -= self.compute_queue[0][1]   # 减去任务所需的时间，并从队列中移除。
                    t_task = self.compute_queue.pop(0)
                    self.task_success += 1     # 增加成功完成任务的计数器self.task_success
                    self.time_counter += ((t - t_task[0].start_time) + (1 - total_time))
                    # for k in range(len(self.replay_buffer)):  # 实质上本地计算的不参与buffer训练
                    #     if self.replay_buffer[k][0] == float(t_task[0].name[len(self.name) + 1:]):
                    #         # print("本地找到该任务")
                    #         self.replay_buffer[k][-1] = 1  # 任务完成状况
                    #         break
                else:
                    self.compute_queue[0][1] -= 1   # 减去这个时间片，在下个时间片执行
                    break
            else:
                break

    def send_to_uav(self, t):
        total_time = 3   # 表示每个任务需要发送给目标系统的时间  模拟并行发送
        while total_time > 0:
            if len(self.send_sfn_queue) > 0:
                if self.send_sfn_queue[0][1] <= total_time:
                    total_time -= self.send_sfn_queue[0][1]
                    t_task = self.send_sfn_queue.pop(0)

                    # 获取任务ID
                    task_id = float(t_task[0].name[len(self.name) + 1:])
                    if self.bind_sfn is not None:
                        self.send_sfn_success += 1
                        reward = round((t - t_task[0].start_time) + (3 - total_time), 2)    # 为什么不是3-total_time
                        # print(f"reward: {reward}  t - t_task[0].start_time: {t - t_task[0].start_time}  3 - total_time {3 - total_time} ")
                        self.bind_sfn.receive_task(t, t_task[0], self.name)
                    else:
                        # print("没有绑定sfn,任务丢弃")
                        self.send_sfn_drop += 1
                        reward = 100
                    for i in range(len(self.replay_buffer)):
                        # print(f'self.replay_buffer[i][0]:{self.replay_buffer[i][0]}   float(t_task[0].name[len(self.name) + 1:]){float(t_task[0].name[len(self.name) + 1:])}')
                        if self.replay_buffer[i][0] == float(t_task[0].name[len(self.name) + 1:]) or self.mode != "PRDMAPPO" :
                            energy_reward = t_task[1] * config.p_tx
                            # print("边缘侧能耗，" , energy_reward)
                            self.energy_comsumpution += energy_reward
                            self.success_energy_num += 1
                            delay_reward = reward
                            # reward = config.r_w * delay_reward + (1 - config.r_w) * energy_reward 
                            reward = delay_reward
                            # print(f"uav: delay_reward:{delay_reward}  energy_reward:{energy_reward}  reward:{reward}")
                            self.task_rewards[task_id] = reward
                    
                            # 直接在本地处理的任务即可更新回合奖励
                            self.update_episode_rewards(task_id, reward)
                            self.replay_buffer[i][self.N_STATE + 2] = -reward
                            
                            self.replay_buffer[i][-1] = 1  # 任务完成状况
                            break

                else:
                    self.send_sfn_queue[0][1] -= total_time
                    total_time = 0
            else:
                break

    def send_to_hap(self, t):
        total_time = 5
        while total_time > 0:
            if len(self.send_cloud_queue) > 0:
                if self.send_cloud_queue[0][1] <= total_time:
                    total_time -= self.send_cloud_queue[0][1]
                    t_task = self.send_cloud_queue.pop(0)
                    self.send_cloud_success += 1
                    self.cloud.receive_task(t_task[0], t)
                    tmp = t_task[0].name[t_task[0].name.find('_') + 1:]
                    for k in range(len(self.replay_buffer)):
                        if self.replay_buffer[k][0] == float(tmp[tmp.find('_') + 1:]):
                            self.replay_buffer[k][-1] = 1  # 任务完成状况
                else:
                    self.send_cloud_queue[0][1] -= total_time
                    total_time = 0
            else:
                break

    def get_action(self, state_policy, greedy=False):
        if np.random.rand() < 0.1:
            action = random.randint(0, self.N_ACTIONS-1)
            return action
        with torch.no_grad():
            state_policy = np.array(state_policy)
            # 将输入状态转换为 torch 张量，并调整维度
            state_policy = torch.from_numpy(state_policy).float().to(self.device).unsqueeze(0)

            # 获取动作分布
            dist = self.car_policy_network(state_policy).squeeze(0)

            if greedy:
                # 使用贪心策略选择动作
                action = dist.argmax().detach().cpu().item()
            else:
                # # 取动作分布的倒数并归一化
                # inv_dist = 1.0 / (dist + 1e-6)  # 加1e-6防止除零
                # inv_dist /= inv_dist.sum()  # 归一化
                # # print(f"dist:{dist}  inv_dist:{inv_dist}")
                # # 使用倒数分布进行采样，优先选择原本概率最小的动作
                # epsilon = 0
                # if random.random() < epsilon:
                #     # 随机探索
                #     action = random.randint(0, self.N_ACTIONS-1)
                # else:
                action = Categorical(dist).sample().detach().cpu().item()

            # 计算动作的 one-hot 编码
            one_hot_action = np.zeros(self.N_ACTIONS)
            one_hot_action[action] = 1

            probs = Categorical(dist)
            action_logprob = probs.log_prob(torch.tensor(action).to(self.device)).item()
            # print("dist_shape:", dist.shape)
            # print("buffer.shape",self.replay_buffer.shape)
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][3+self.N_STATE*2 : 3 + self.N_STATE * 2 + self.N_ACTIONS] = dist.cpu().numpy()
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][3+self.N_STATE*2+self.N_ACTIONS : 3+self.N_STATE*2+self.N_ACTIONS*2] = one_hot_action
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][3+self.N_STATE*2+self.N_ACTIONS*2] = action_logprob
            # # 保存动作分布、动作的 log 概率和 one-hot 编码
            # self.buffer.probs[self.index, self.id, :] = dist_probs_numpy
            # self.buffer.logprobs[self.index, self.id, :] = action_logprob_numpy
            # self.buffer.one_hot_actions[self.index, self.id, :] = one_hot_action
            return action

    def wait_time(self, t, object):
        if object == 'local':
            if len(self.compute_queue) == 0:
                return 0
            else:
                return self.compute_queue[-1][-1] - t
        elif object == 'cloud':
            if len(self.send_cloud_queue) == 0:
                return 0
            else:
                return self.send_cloud_queue[-1][-1] - t
        elif object == 'sfn':
            if len(self.send_sfn_queue) == 0:
                return 0
            else:
                return self.send_sfn_queue[-1][-1] - t

    def memory_check(self, task):
        total_res = 0
        for i in range(len(self.compute_queue)):
            total_res += self.compute_queue[i][0].required_res
        if total_res + task.required_res <= self.c:
            return True
        else:
            return False

    def scan_uav(self, t):
        cur_pos = self.route[t]
        for bp in config.base_pos:
            if math.sqrt(math.pow(cur_pos[0] - bp[0], 2) + math.pow(cur_pos[1] - bp[1], 2)) <= config.base_cover:   # 计算当前位置和每个基站的距离是否<=基站的覆盖范围
                if self.bind_sfn is None:
                    self.bind_sfn = self.sfns[bp]
                    if self.name not in self.bind_sfn.bind_car:  # bind_car是SFN类中的一个属性，用于存储绑定到该SFN的移动边缘计算节点的列表。当一个 MFN 被绑定到一个 SFN 时，它会将自己的名称添加到对应SFN的 bind_car列表中。这样做有助于SFN跟踪绑定到自己的所有MFN，并管理与这些节点之间的通信和任务调度。
                        self.bind_sfn.bind_car.append(self.name)
                return
        if self.bind_sfn is not None:   #  因为如果在基站范围内就已经return了，不会到这一步，所以此时绑定的， 超出已经绑定SFN的覆盖范围。
            self.bind_sfn.bind_car.remove(self.name)   # 将当前节点从绑定的车辆列表中移除
            self.bind_sfn = None
            self.send_sfn_drop += len(self.send_sfn_queue)   # 将发送给该 SFN 的任务数量累加到当前节点的 send_sfn_drop 属性中。
            # print(f"因为没有绑定到 SFN, 任务丢弃{len(self.send_sfn_queue)}条")
            while len(self.send_sfn_queue) > 0:
                t_task = self.send_sfn_queue.pop()
                # 获取任务ID
                task_id = float(t_task[0].name[len(self.name) + 1:])
                for i in range(len(self.replay_buffer)):
                    if float(t_task[0].name[len(self.name) + 1:]) == self.replay_buffer[i][0]:   # 用于检查当前处理的任务是否与 replay_buffer 中的某个任务匹配
                        self.replay_buffer[i][self.N_STATE+2] = -100 # 代表放弃执行或者其他特定的处理方式。
                        self.task_rewards[task_id] = 100
                    
                        # 直接在本地处理的任务即可更新回合奖励
                        self.update_episode_rewards(task_id, 100)
                        break

    def offload(self, t, task, state):
        # 添加调试信息
        # print(f"Entering offload function:")
        # print(f"task.name: {task.name}")
        # print(f"self.name: {self.name}")
        # print(f"task.name[len(self.name) + 1:]: {task.name[len(self.name) + 1:]}")
        if self.mode == "PRDMAPPO":
            # print("index:", self.index)
            if self.index > config.MEMORY_CAPACITY:
                epsilon = max(0.1, 0.1 - 0.0001 * self.index)  # 随时间衰减
                if random.random() < epsilon:
                    action = random.randint(0, self.N_ACTIONS-1)
                else:
                    action = self.get_action(state)
            elif self.N_ACTIONS == 2:
                if self.comp_mfn_time(t, state) >= task.task_size / config.g2u_band:
                    action = 0
                else:
                    action = 1
            else:
                action = random.randint(0, self.N_ACTIONS-1)   
            # print(f"self.name:{self.name}  task.name:{task.name}")
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][0] = float(task.name[len(self.name) + 1:])
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][1:1+self.N_STATE] = np.hstack(state)
            self.replay_buffer[self.index % config.MEMORY_CAPACITY][self.N_STATE+1] = action
            if self.index > 0:
                self.replay_buffer[(self.index - 1) % config.MEMORY_CAPACITY][self.N_STATE+3:3+self.N_STATE*2] = np.hstack(state)
            self.index += 1
            return action
        elif self.mode == "random":
            action = random.randint(0, self.N_ACTIONS-1)
            return action
        elif self.mode == "Heuristic":
            if self.N_ACTIONS == 2:
                if self.comp_mfn_time(t, state) >= task.task_size / config.g2u_band:
                    action = 1
                else:
                    action = 2
            else:      
                if self.comp_mfn_time(t, state) >= task.task_size / config.g2u_band:
                    if random.random() < 0.2:
                        action = 0
                    else:
                        action = 1
                else:
                    action = 2
                return action
        else:
            if self.N_ACTIONS == 2:
                if self.bind_sfn != None:
                    action = 0
                else:
                    action = 1
                return action
            else:
                if self.bind_sfn != None:
                    if random.random() < 0.05:  
                        action = 0
                    else:
                        action = 1
                else:
                    action = 2
                return action

    def comp_mfn_time(self, t, state):   # 计算移动目标设备（Mobile Fog Node，MFN）到达目标位置所需的时间
        if math.sqrt(math.pow(self.route[t][0] - self.bind_sfn.pos[0], 2) + math.pow(self.route[t][1] - self.bind_sfn.pos[1], 2)) <= math.sqrt(math.pow(self.route[t-1][0] -  self.bind_sfn.pos[0], 2) + math.pow(self.route[t-1][1] -  self.bind_sfn.pos[1], 2)):
            return 100
        else:
            dis = 100
            for d in config.base_pos:
                dis = min(dis, math.sqrt(math.pow(self.route[t][0] - d[0], 2) + math.pow(self.route[t][1] - d[1], 2)))
            return dis / 14

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
    

    # 车辆侧更新
    def car_update(self, episode):
        # convert list to tensor
        car_old_states = torch.FloatTensor(np.array(self.car_buffer.states)).to(self.device)
        car_old_actions = torch.FloatTensor(np.array(self.car_buffer.actions)).to(self.device)
        car_old_one_hot_actions = torch.FloatTensor(np.array(self.car_buffer.one_hot_actions)).to(self.device)
        car_old_probs = torch.FloatTensor(np.array(self.car_buffer.probs)).to(self.device)
        car_old_logprobs = torch.FloatTensor(np.array(self.car_buffer.logprobs)).to(self.device)
        car_rewards = torch.FloatTensor(np.array(self.car_buffer.rewards)).to(self.device)
        car_dones = torch.FloatTensor(np.array(self.car_buffer.dones)).long().to(self.device)
        # 获取样本总数
        # num_samples = car_old_states.size(0)
        # 随机生成索引
        # indices = torch.randperm(num_samples)[:56]
        # # 使用随机索引重新排列数据
        # car_old_states = car_old_states[indices]
        # car_old_actions = car_old_actions[indices]
        # car_old_one_hot_actions = car_old_one_hot_actions[indices]
        # car_old_probs = car_old_probs[indices]
        # car_old_logprobs = car_old_logprobs[indices]
        # car_rewards = car_rewards[indices]
        # car_dones = car_dones[indices]


        # print("actions:", car_old_actions.shape)  # 这里是0
        # print(car_old_actions)
        # print("states:", car_old_states.shape)
        # print(car_old_states)
        # print("probs:", car_old_probs.shape)
        # print(car_old_probs)
        # print("one_hot_actions:", car_old_one_hot_actions.shape)
        # print(car_old_one_hot_actions)
        # print("logprobs:", car_old_logprobs.shape)
        # print(car_old_logprobs)
        # print("reward:", car_rewards.shape)
        # print(car_rewards)
        # print("dones:", car_dones.shape)
        # print(car_dones)


        # 配置日志
        logging.basicConfig(filename='output_MFN_action_log.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s')

        # 设置计数器


        # 打印并记录信息
        logging.info(f"--- Log Entry {self.log_count} ---")
        logging.info(f"actions shape: {car_old_actions.shape}")
        logging.info(f"actions:\n{car_old_actions}")
        # logging.info(f"states shape: {car_old_states.shape}")
        # logging.info(f"states:\n{car_old_states}")
        # logging.info(f"probs shape: {car_old_probs.shape}")
        # logging.info(f"probs:\n{car_old_probs}")
        # logging.info(f"one_hot_actions shape: {car_old_one_hot_actions.shape}")
        # logging.info(f"one_hot_actions:\n{car_old_one_hot_actions}")
        # logging.info(f"logprobs shape: {car_old_logprobs.shape}")
        # logging.info(f"logprobs:\n{car_old_logprobs}")
        logging.info(f"reward shape: {car_rewards.shape}")
        logging.info(f"reward:\n{car_rewards}")
        # logging.info(f"dones shape: {car_dones.shape}")
        # logging.info(f"dones:\n{car_dones}")
        # logging.info(f"--- End of Log Entry {count} ---")

        # 增加计数
        self.log_count += 1

        car_Values_old, car_Q_values_old, car_weights_value_old = self.car_critic_network_old(car_old_states,
                                                                                              car_old_probs.squeeze(
                                                                                                  -2),
                                                                                              car_old_one_hot_actions)
        car_Values_old = car_Values_old.reshape(-1, self.car_num, self.car_num)

        car_Q_value_target = self.nstep_returns(car_Q_values_old, car_rewards, car_dones).detach()

        car_value_loss_batch = 0
        car_policy_loss_batch = 0
        car_entropy_batch = 0
        car_value_weights_batch = None
        car_grad_norm_value_batch = 0
        car_grad_norm_policy_batch = 0
        car_agent_groups_over_episode_batch = 0
        car_avg_agent_group_over_episode_batch = 0

        # torch.autograd.set_detect_anomaly(True)
        # Optimize policy for n epochs
        for _ in range(self.n_epochs):

            car_Value, car_Q_value, car_weights_value = self.car_critic_network(car_old_states,
                                                                                car_old_probs.squeeze(-2),
                                                                                car_old_one_hot_actions)
            car_Value = car_Value.reshape(-1, self.car_num, self.car_num)

            car_advantage, car_masking_advantage = self.calculate_advantages_based_on_exp(car_Value, car_rewards,
                                                                                          car_dones,
                                                                                          car_weights_value,
                                                                                          episode)

            if "threshold" in self.experiment_type:
                car_agent_groups_over_episode = torch.sum(torch.sum(car_masking_advantage.float(), dim=-2), dim=0) / \
                                                car_masking_advantage.shape[0]
                car_avg_agent_group_over_episode = torch.mean(car_agent_groups_over_episode)
                car_agent_groups_over_episode_batch += car_agent_groups_over_episode
                car_avg_agent_group_over_episode_batch += car_avg_agent_group_over_episode

            dists = self.car_policy_network(car_old_states)
            probs = Categorical(dists.squeeze(0))
            # print("car_old_actions.shape:", car_old_actions.shape)
            # print("car_old_actions:", car_old_actions)
            logprobs = probs.log_prob(car_old_actions)

            car_critic_loss_1 = F.smooth_l1_loss(car_Q_value, car_Q_value_target)
            car_critic_loss_2 = F.smooth_l1_loss(
                torch.clamp(car_Q_value, car_Q_values_old - self.value_clip, car_Q_values_old + self.value_clip),
                car_Q_value_target)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - car_old_logprobs)
            # Finding Surrogate Loss
            surr1 = ratios * car_advantage.detach()
            surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * car_advantage.detach()

            # final loss of clipped objective PPO
            entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=2))
            car_policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen * entropy

            entropy_weights = -torch.mean(
                torch.sum(car_weights_value * torch.log(torch.clamp(car_weights_value, 1e-10, 1.0)), dim=2))
            car_critic_loss = torch.max(car_critic_loss_1, car_critic_loss_2)

            # take gradient step
            self.car_critic_optimizer.zero_grad()
            car_critic_loss.backward()
            grad_norm_value = torch.nn.utils.clip_grad_norm_(self.car_critic_network.parameters(),
                                                             self.grad_clip_critic)
            self.car_critic_optimizer.step()

            self.car_policy_optimizer.zero_grad()
            car_policy_loss.backward()
            grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.car_policy_network.parameters(),
                                                              self.grad_clip_actor)
            self.car_policy_optimizer.step()

            if self.dictionary["load_models"]:
                print("LOADING MODELS")
                # Loading models
                self.car_critic_network_old.load_state_dict(
                    torch.load(self.dictionary["critic_dir"] + "car_critic_network.pt"))
                self.car_critic_network.load_state_dict(
                    torch.load(self.dictionary["critic_dir"] + "car_critic_network.pt"))
                self.car_policy_network_old.load_state_dict(
                    torch.load(self.dictionary["actor_dir"] + "car_actor_network.pt"))
                self.car_policy_network.load_state_dict(
                    torch.load(self.dictionary["actor_dir"] + "car_actor_network.pt"))

            car_value_loss_batch += car_critic_loss
            car_policy_loss_batch += car_policy_loss
            car_entropy_batch += entropy
            car_grad_norm_value_batch += grad_norm_value
            car_grad_norm_policy_batch += grad_norm_policy
            if car_value_weights_batch is None:
                car_value_weights_batch = torch.zeros_like(car_weights_value)
            car_value_weights_batch += car_weights_value.detach()

        # Copy new weights into old policy
        self.car_policy_network_old.load_state_dict(self.car_policy_network.state_dict())

        # Copy new weights into old critic
        self.car_critic_network_old.load_state_dict(self.car_critic_network.state_dict())

        torch.save(self.car_critic_network.state_dict(), self.dictionary["critic_dir"] + "car_critic_network.pt")
        torch.save(self.car_policy_network.state_dict(), self.dictionary["actor_dir"] + "car_actor_network.pt")

        # clear buffer
        # self.car_buffer.clear()

        car_value_loss_batch /= self.n_epochs
        car_policy_loss_batch /= self.n_epochs
        car_entropy_batch /= self.n_epochs
        car_grad_norm_value_batch /= self.n_epochs
        car_grad_norm_policy_batch /= self.n_epochs
        car_value_weights_batch /= self.n_epochs
        car_agent_groups_over_episode_batch /= self.n_epochs
        car_avg_agent_group_over_episode_batch /= self.n_epochs

        # if "prd" in self.experiment_type:
        #     num_relevant_agents_in_relevant_set = self.car_relevant_set * car_masking_advantage
        #     num_non_relevant_agents_in_relevant_set = self.car_non_relevant_set * car_masking_advantage
        #     true_negatives = self.car_non_relevant_set * (1 - car_masking_advantage)
        # else:
        #     num_relevant_agents_in_relevant_set = None
        #     num_non_relevant_agents_in_relevant_set = None
        #     true_negatives = None
        #
        # threshold = self.select_above_threshold

    # def car_update(self, episode):
    #     # convert list to tensor
    #     car_old_states = torch.FloatTensor(np.array(self.car_buffer.states)).to(self.device)
    #     car_old_actions = torch.FloatTensor(np.array(self.car_buffer.actions)).to(self.device)
    #     car_old_one_hot_actions = torch.FloatTensor(np.array(self.car_buffer.one_hot_actions)).to(self.device)
    #     car_old_probs = torch.FloatTensor(np.array(self.car_buffer.probs)).to(self.device)
    #     car_old_logprobs = torch.FloatTensor(np.array(self.car_buffer.logprobs)).to(self.device)
    #     car_rewards = torch.FloatTensor(np.array(self.car_buffer.rewards)).to(self.device)
    #     car_dones = torch.FloatTensor(np.array(self.car_buffer.dones)).long().to(self.device)
    #
    #     # 获取样本总数
    #     num_samples = car_old_states.size(0)
    #
    #     # 优化策略的迭代次数
    #     for _ in range(self.n_epochs):
    #         # 随机生成索引
    #         indices = torch.randperm(num_samples)[:56]
    #
    #         # 使用随机索引重新排列数据
    #         sampled_states = car_old_states[indices]
    #         sampled_actions = car_old_actions[indices]
    #         sampled_one_hot_actions = car_old_one_hot_actions[indices]
    #         sampled_probs = car_old_probs[indices]
    #         sampled_logprobs = car_old_logprobs[indices]
    #         sampled_rewards = car_rewards[indices]
    #         sampled_dones = car_dones[indices]
    #
    #         car_Values_old, car_Q_values_old, car_weights_value_old = self.car_critic_network_old(
    #             sampled_states,
    #             sampled_probs.squeeze(-2),
    #             sampled_one_hot_actions
    #         )
    #         car_Values_old = car_Values_old.reshape(-1, self.car_num, self.car_num)
    #
    #         car_Q_value_target = self.nstep_returns(car_Q_values_old, sampled_rewards, sampled_dones).detach()
    #
    #         car_value_loss_batch = 0
    #         car_policy_loss_batch = 0
    #         car_entropy_batch = 0
    #         car_value_weights_batch = None
    #         car_grad_norm_value_batch = 0
    #         car_grad_norm_policy_batch = 0
    #         car_agent_groups_over_episode_batch = 0
    #         car_avg_agent_group_over_episode_batch = 0
    #
    #         # torch.autograd.set_detect_anomaly(True)
    #         # Optimize policy for n epochs
    #         car_Value, car_Q_value, car_weights_value = self.car_critic_network(sampled_states,
    #                                                                             sampled_probs.squeeze(-2),
    #                                                                             sampled_one_hot_actions)
    #         car_Value = car_Value.reshape(-1, self.car_num, self.car_num)
    #
    #         car_advantage, car_masking_advantage = self.calculate_advantages_based_on_exp(car_Value, sampled_rewards,
    #                                                                                       sampled_dones,
    #                                                                                       car_weights_value,
    #                                                                                       episode)
    #
    #         if "threshold" in self.experiment_type:
    #             car_agent_groups_over_episode = torch.sum(torch.sum(car_masking_advantage.float(), dim=-2), dim=0) / \
    #                                             car_masking_advantage.shape[0]
    #             car_avg_agent_group_over_episode = torch.mean(car_agent_groups_over_episode)
    #             car_agent_groups_over_episode_batch += car_agent_groups_over_episode
    #             car_avg_agent_group_over_episode_batch += car_avg_agent_group_over_episode
    #
    #         dists = self.car_policy_network(sampled_states)
    #         probs = Categorical(dists.squeeze(0))
    #         logprobs = probs.log_prob(sampled_actions)
    #
    #         car_critic_loss_1 = F.smooth_l1_loss(car_Q_value, car_Q_value_target)
    #         car_critic_loss_2 = F.smooth_l1_loss(
    #             torch.clamp(car_Q_value, car_Q_values_old - self.value_clip, car_Q_values_old + self.value_clip),
    #             car_Q_value_target)
    #
    #         # Finding the ratio (pi_theta / pi_theta__old)
    #         ratios = torch.exp(logprobs - sampled_logprobs)
    #         # Finding Surrogate Loss
    #         surr1 = ratios * car_advantage.detach()
    #         surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * car_advantage.detach()
    #
    #         # final loss of clipped objective PPO
    #         entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=2))
    #         car_policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen * entropy
    #
    #         entropy_weights = -torch.mean(
    #             torch.sum(car_weights_value * torch.log(torch.clamp(car_weights_value, 1e-10, 1.0)), dim=2))
    #         car_critic_loss = torch.max(car_critic_loss_1, car_critic_loss_2)
    #
    #         # take gradient step
    #         self.car_critic_optimizer.zero_grad()
    #         car_critic_loss.backward()
    #         grad_norm_value = torch.nn.utils.clip_grad_norm_(self.car_critic_network.parameters(),
    #                                                          self.grad_clip_critic)
    #         self.car_critic_optimizer.step()
    #
    #         self.car_policy_optimizer.zero_grad()
    #         car_policy_loss.backward()
    #         grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.car_policy_network.parameters(),
    #                                                           self.grad_clip_actor)
    #         self.car_policy_optimizer.step()
    #
    #         if self.dictionary["load_models"]:
    #             print("LOADING MODELS")
    #             # Loading models
    #             self.car_critic_network_old.load_state_dict(
    #                 torch.load(self.dictionary["critic_dir"] + "car_critic_network.pt"))
    #             self.car_critic_network.load_state_dict(
    #                 torch.load(self.dictionary["critic_dir"] + "car_critic_network.pt"))
    #             self.car_policy_network_old.load_state_dict(
    #                 torch.load(self.dictionary["actor_dir"] + "car_actor_network.pt"))
    #             self.car_policy_network.load_state_dict(
    #                 torch.load(self.dictionary["actor_dir"] + "car_actor_network.pt"))
    #
    #         car_value_loss_batch += car_critic_loss
    #         car_policy_loss_batch += car_policy_loss
    #         car_entropy_batch += entropy
    #         car_grad_norm_value_batch += grad_norm_value
    #         car_grad_norm_policy_batch += grad_norm_policy
    #         if car_value_weights_batch is None:
    #             car_value_weights_batch = torch.zeros_like(car_weights_value)
    #         car_value_weights_batch += car_weights_value.detach()
    #
    #     # Copy new weights into old policy
    #     self.car_policy_network_old.load_state_dict(self.car_policy_network.state_dict())
    #
    #     # Copy new weights into old critic
    #     self.car_critic_network_old.load_state_dict(self.car_critic_network.state_dict())
    #
    #     torch.save(self.car_critic_network.state_dict(), self.dictionary["critic_dir"] + "car_critic_network.pt")
    #     torch.save(self.car_policy_network.state_dict(), self.dictionary["actor_dir"] + "car_actor_network.pt")
    #
    #     # clear buffer
    #     self.car_buffer.clear()
    #
    #     car_value_loss_batch /= self.n_epochs
    #     car_policy_loss_batch /= self.n_epochs
    #     car_entropy_batch /= self.n_epochs
    #     car_grad_norm_value_batch /= self.n_epochs
    #     car_grad_norm_policy_batch /= self.n_epochs
    #     car_value_weights_batch /= self.n_epochs
    #     car_agent_groups_over_episode_batch /= self.n_epochs
    #     car_avg_agent_group_over_episode_batch /= self.n_epochs