import os
import random
import time
import numpy as np
from torch import optim
from torch.distributions import Categorical

from HAP import HAP
from UAV_load_balance import UAV
import torch
from utils import Task
import model
import config
from Ground_Car import Ground
from Ground_Station import GroundStation
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import default_device

class System:
    # state=9, action=5, sfn_offload_mode='A3C', mfn_offload_mode='DQN'
    def __init__(self,  n_state, n_action, n_mfn, sfn_offload_mode, mfn_offload_mode, task_gen_rate, dictionary):

        self.offload_mode = sfn_offload_mode
        self.device = default_device()
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
        self.gd_num = n_mfn
        self.num_agents = 9
        self.num_actions = n_action

        self.car_num = 12  # 6 + 6 
        self.car_actions = 3
        self.ground_actions = 2
        self.car_state = 12
        # self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
        self.dictionary = dictionary
        self.task_generation = task_gen_rate
        self.state_num = 20
        self.buffer = model.RolloutBuffer(config.MEMORY_CAPACITY_AC, self.num_agents, self.state_num, self.num_actions)
        # self.comet_ml = None
        # if self.save_comet_ml_plot:
        #     self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I", project_name=dictionary["test_num"])
        #     self.comet_ml.log_parameters(dictionary)
        self.cloud = HAP()
        self.sfn_offload_mode = sfn_offload_mode
        self.mfn_offload_mode = mfn_offload_mode


#   前面都网络都是边缘服务器智能体，后面是车辆智能体群域

        self.car_buffer = model.RolloutBuffer(config.MEMORY_CAPACITY, self.car_num, self.car_state, self.car_actions)
        self.station_buffer = model.RolloutBuffer(config.MEMORY_CAPACITY, self.car_num, self.car_state, self.ground_actions)
        self.sfn1 = UAV(0, [150, 150], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn2 = UAV(1, [150, 450], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn3 = UAV(2, [150, 750], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn4 = UAV(3, [450, 150], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn5 = UAV(4, [450, 450], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn6 = UAV(5, [450, 750], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn7 = UAV(6, [750, 150], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn8 = UAV(7, [750, 450], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfn9 = UAV(8, [750, 750], sfn_offload_mode,  self.cloud, dictionary, self.buffer)
        self.sfns = [self.sfn1, self.sfn2, self.sfn3, self.sfn4, self.sfn5, self.sfn6, self.sfn7, self.sfn8, self.sfn9]
        self.sfn1.sfns = self.sfns
        self.sfn2.sfns = self.sfns
        self.sfn3.sfns = self.sfns
        self.sfn4.sfns = self.sfns
        self.sfn5.sfns = self.sfns
        self.sfn6.sfns = self.sfns
        self.sfn7.sfns = self.sfns
        self.sfn8.sfns = self.sfns
        self.sfn9.sfns = self.sfns
        sfns = {(150, 150): self.sfn1, (150, 450): self.sfn2, (150, 750): self.sfn3, (450, 150): self.sfn4, (450, 450): self.sfn5, (450, 750): self.sfn6, (750, 150): self.sfn7, (750, 450): self.sfn8, (750, 750): self.sfn9}
        self.agent_num = len(self.sfns)
        self.mfn_offload_mode = mfn_offload_mode
        self.mfns = list()
        for i in range(1, 1 + n_mfn):
            self.mfns.append(Ground(i-1,random.randint(config.f_range_mfn[0], config.f_range_mfn[1]),
                                 random.randint(config.c_range_mfn[0], config.c_range_mfn[1]),
                                 'car_' + str(i), sfns, self.cloud, mfn_offload_mode, task_gen_rate, dictionary, self.car_buffer, n_mfn))
            # else:
            #     self.mfns.append(GroundStation(i-1,random.randint(config.f_range_mfn[0], config.f_range_mfn[1]),
            #                      random.randint(config.c_range_mfn[0], config.c_range_mfn[1]),
            #                      'gro_' + str(i-6), sfns, self.cloud, mfn_offload_mode, task_gen_rate, dictionary, self.station_buffer))
        all_cars = {car.name: car for car in self.mfns}
        # 将 all_cars 放入每个 Ground 的 dictionary
        for car in self.mfns:
            car.dictionary["all_cars"] = all_cars  
        for sfn in self.sfns:
            sfn.dictionary["all_cars"] = all_cars

    def start(self, time_long):   # 12个小时
        print("服务器卸载算法为：", self.sfn_offload_mode)
        print("车辆卸载算法为：", self.mfn_offload_mode)

        # 初始化回合奖励收集 
        for sfn in self.sfns:
            if not hasattr(sfn, 'episode_rewards'):
                sfn.episode_rewards = []
                sfn.current_episode_reward = 0
                sfn.episode_task_count = 0
                sfn.episode_length = 50  # 每50个任务为一个回合
        
        for mfn in self.mfns:
            if not hasattr(mfn, 'episode_rewards'):
                mfn.episode_rewards = []
                mfn.current_episode_reward = 0
                mfn.episode_task_count = 0
                mfn.episode_length = 50  # 每50个任务为一个回合

        for t in tqdm(range(3600 * time_long + 2000)):
            for mfn in self.mfns:
                if t > 0 and (t % (self.update_ppo_agent-1)) == 0:
                    
                    # start = time.time()
                    self.car_render_buffer(mfn.id, mfn.replay_buffer)
                    
                    # else:
                    #     self.station_render_buffer(mfn.id-6, mfn.replay_buffer)
                    
                    # end = time.time()
                    # print("一辆车提交任务所需时间：", end - start)
                mfn.run(t)
            for sfn in self.sfns:
                if t > 0 and (t % (config.MEMORY_CAPACITY_AC - 1)) == 0:   # 需要每个边缘服务器帮我提交信息。
                    # start = time.time()
                    self.render_buffer(sfn.id, sfn.replay_buffer)
                    # end = time.time()
                    # print("一个边缘服务器提交任务所需的时间：", end - start)
                sfn.run(t)
            self.cloud.run(t)

            # 在模拟结束时生成收敛性分析图
            if t == 3600 * time_long + 1999:
                self.plot_convergence()
            
    

    def render_buffer(self, id, replay_buffer):
        self.buffer.rewards[:, id] = replay_buffer[:, self.state_num + 2]
        self.buffer.dones[:, id] = replay_buffer[:, -1]
        self.buffer.states[:, id, :] = replay_buffer[:, 1:self.state_num + 1]
        self.buffer.actions[:, id] = replay_buffer[:, self.state_num+1]
        self.buffer.probs[:, id ,:] = replay_buffer[:, 3+self.state_num*2:3+self.state_num*2+self.num_actions]
        self.buffer.logprobs[:, id] = replay_buffer[:, 3+self.state_num*2+self.num_actions*2]
        self.buffer.one_hot_actions[:, id, :] = replay_buffer[:, 3+self.state_num*2+self.num_actions:3+self.state_num*2+self.num_actions*2]

    def car_render_buffer(self, id, replay_buffer):
        self.car_buffer.rewards[:, id] = replay_buffer[:, self.car_state + 2]
        self.car_buffer.dones[:, id] = replay_buffer[:, -1]
        self.car_buffer.states[:, id, :] = replay_buffer[:, 1:self.car_state + 1]
        self.car_buffer.actions[:, id] = replay_buffer[:, self.car_state + 1]
        self.car_buffer.probs[:, id, :] = replay_buffer[:, 3 + self.car_state * 2:3 + self.car_state * 2 + self.car_actions]
        self.car_buffer.logprobs[:, id] = replay_buffer[:, 3 + self.car_state * 2 + self.car_actions * 2]
        self.car_buffer.one_hot_actions[:, id, :] = replay_buffer[:,
                                                3 + self.car_state * 2 + self.car_actions:3 + self.car_state * 2 + self.car_actions * 2]

    def station_render_buffer(self, id, replay_buffer):
        self.station_buffer.rewards[:, id] = replay_buffer[:, self.car_state + 2]
        self.station_buffer.dones[:, id] = replay_buffer[:, -1]
        self.station_buffer.states[:, id, :] = replay_buffer[:, 1:self.car_state + 1]
        self.station_buffer.actions[:, id] = replay_buffer[:, self.car_state + 1]
        self.station_buffer.probs[:, id, :] = replay_buffer[:, 3 + self.car_state * 2:3 + self.car_state * 2 + self.ground_actions]
        self.station_buffer.logprobs[:, id] = replay_buffer[:, 3 + self.car_state * 2 + self.ground_actions * 2]
        self.station_buffer.one_hot_actions[:, id, :] = replay_buffer[:,
                                                3 + self.car_state * 2 + self.ground_actions:3 + self.car_state * 2 + self.ground_actions * 2]

    def collect_episode_rewards(self):
        """收集车辆和边缘服务器的回合奖励"""
        # 分别存储SFN(边缘服务器)和MFN(车辆)的奖励
        sfn_rewards = []
        mfn_rewards = []
        
        # 收集所有边缘服务器的奖励
        for sfn in self.sfns:
            if hasattr(sfn, 'episode_rewards') and len(sfn.episode_rewards) > 0:
                sfn_rewards.append(sfn.episode_rewards)
        
        # 收集所有车辆的奖励
        for mfn in self.mfns:
            if hasattr(mfn, 'episode_rewards') and len(mfn.episode_rewards) > 0:
                mfn_rewards.append(mfn.episode_rewards)
        
        # 如果有数据则处理
        result = {}
        
        # 处理边缘服务器奖励
        if sfn_rewards:
            # 找出最小回合数，确保所有数组长度一致
            min_episodes_sfn = min(len(rewards) for rewards in sfn_rewards)
            # 对齐所有数组长度
            aligned_sfn = [rewards[:min_episodes_sfn] for rewards in sfn_rewards]
            # 计算每个回合的平均值
            sfn_avg = []
            for ep in range(min_episodes_sfn):
                episode_sum = sum(rewards[ep] for rewards in aligned_sfn)
                episode_avg = episode_sum / len(aligned_sfn)
                sfn_avg.append(episode_avg)
            result["SFN"] = sfn_avg
        
        # 处理车辆奖励
        if mfn_rewards:
            # 找出最小回合数，确保所有数组长度一致
            min_episodes_mfn = min(len(rewards) for rewards in mfn_rewards)
            # 对齐所有数组长度
            aligned_mfn = [rewards[:min_episodes_mfn] for rewards in mfn_rewards]
            # 计算每个回合的平均值
            mfn_avg = []
            for ep in range(min_episodes_mfn):
                episode_sum = sum(rewards[ep] for rewards in aligned_mfn)
                episode_avg = episode_sum / len(aligned_mfn)
                mfn_avg.append(episode_avg)
            result["MFN"] = mfn_avg
        
        return result

    def plot_convergence(self, save_path=None):
        """绘制车辆和边缘服务器的收敛性分析图"""
        rewards = self.collect_episode_rewards()
        
        if not rewards:
            print("没有收集到任何奖励数据用于绘图")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 应用平滑处理函数
        def smooth_rewards(rewards_array, window=5):
            """对奖励序列应用滑动平均平滑"""
            if len(rewards_array) <= window:
                return rewards_array
            return np.convolve(rewards_array, np.ones(window)/window, mode='valid')
        
        # 绘制边缘服务器的收敛曲线
        if "SFN" in rewards:
            sfn_rewards = np.array(rewards["SFN"])
            sfn_smoothed = smooth_rewards(sfn_rewards)
            plt.plot(range(1, len(sfn_smoothed) + 1), sfn_smoothed, 
                    color='red', linewidth=2, label=f'Edge Servers ({self.sfn_offload_mode})')
            
            # 保存边缘服务器奖励数据
            if save_path:
                np.save(f'{save_path}_sfn_rewards.npy', sfn_rewards)
        
        # 绘制车辆的收敛曲线
        if "MFN" in rewards:
            mfn_rewards = np.array(rewards["MFN"]) 
            mfn_smoothed = smooth_rewards(mfn_rewards)
            plt.plot(range(1, len(mfn_smoothed) + 1), mfn_smoothed, 
                    color='blue', linewidth=2, linestyle='--', label=f'Vehicles ({self.mfn_offload_mode})')
            
            # 保存车辆奖励数据
            if save_path:
                np.save(f'{save_path}_mfn_rewards.npy', mfn_rewards)
        
        # 设置图表格式
        plt.title(f'Convergence Analysis: {self.sfn_offload_mode}', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Average Reward', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置y轴范围
        plt.ylim([-1000, -200])
        
        # 如果指定了保存路径，则保存图表
        if save_path:
            plt.savefig(f'{save_path}_convergence.png', dpi=300, bbox_inches='tight')
            print(f"收敛性分析图已保存至 {save_path}_convergence.png")
        else:
            save_name = f'{self.sfn_offload_mode}_convergence.png'
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"收敛性分析图已保存至 {save_name}")
        
        plt.close()
    # 车辆侧更新


    # def get_actions(self, greedy=False):
    #     for state_policy in self.buffer.states:
    #         with torch.no_grad():
    #             state_policy = torch.from_numpy(state_policy).float().to(self.device).unsqueeze(0)
    #             dists = self.policy_network_old(state_policy).squeeze(0)
    #             if greedy:
    #                 actions = [dist.argmin().detach().cpu().item() for dist in dists]
    #             else:
    #                 actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]
    #             one_hot_actions = np.zeros((self.num_agents, self.num_actions))
    #             for i, act in enumerate(actions):
    #                 one_hot_actions[i][act] = 1
    #
    #             probs = Categorical(dists)
    #             action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))
    #
    #             self.buffer.actions.append(actions)
    #             self.buffer.probs.append(dists.detach().cpu())
    #             self.buffer.logprobs.append(action_logprob.detach().cpu())
    #             self.buffer.one_hot_actions.append(one_hot_actions)
    #
    #             # return actions

    def print(self):
        total_task_success_mfn = 0
        total_time_success_mfn = 0
        total_end_time_mfn = 0
        total_exec_task_mfn = 0
        total_energy_gd = 0
        total_num_energy = 0
        for mfn in self.mfns:
            total_task_success_mfn += mfn.task_success
            total_time_success_mfn += mfn.time_counter
            total_end_time_mfn += mfn.end_time
            total_exec_task_mfn += mfn.task_exec_count
            total_energy_gd += mfn.energy_comsumpution
            total_num_energy += mfn.success_energy_num
            print('*' * 30)
            print("车辆" + str(mfn.id + 1) + "的卸载情况如下:")
            print('本地计算完成：', mfn.task_success, '本地计算失败：', mfn.task_drop)
            print('本地发送边缘完成：', mfn.send_sfn_success, '本地发送sfn失败：', mfn.send_sfn_drop)
            print('本地发送云完成：', mfn.send_cloud_success)
            print('平均完成时间：', round(mfn.time_counter / mfn.task_success, 2))
            print('能量消耗：', mfn.energy_comsumpution)
        

        print('*' * 30)
        total_time_success_sfn = 0
        total_task_success_sfn = 0
        total_task_cloud = 0
        for sfn in self.sfns:
            total_task_success_sfn += sfn.task_success
            total_time_success_sfn += sfn.time_counter
            total_task_cloud += sfn.task_cloud
            print(f'sfn{sfn.id+1}：计算完成：{sfn.task_success} 总计算字节数：{sfn.task_res} 计算失败：{sfn.task_drop}')

        print('平均完成时间：', round(total_time_success_sfn / total_task_success_sfn, 2))
        print('sfn上Cloud：', total_task_cloud)
        print('*' * 30)
        print('Cloud计算完成：', self.cloud.task_success, '计算失败：', self.cloud.task_drop)
        if self.cloud.task_success == 0:
            print('平均完成时间：', 0)
        else:
            print('平均完成时间：', round(self.cloud.time_counter / self.cloud.task_success))
        print('*' * 30)
        total_task = total_task_success_mfn + total_task_success_sfn + self.cloud.task_success
        total_time = total_time_success_mfn + total_time_success_sfn + self.cloud.time_counter
        print('总的完成任务数：', total_task)
        print('总的完成时间：', total_time)
        success_rate = round(total_task / total_exec_task_mfn, 2)
        print('任务完成率：', success_rate)
        print('平均任务完成时间：', round(total_time / total_task, 2))
        print('加权平均任务完成时间：', success_rate * round(total_time / total_task, 2) + (1 - success_rate) * total_end_time_mfn / total_task)
        print('任务平均能量消耗：', total_energy_gd / total_num_energy)
        print('地面设备平均能耗：' , total_energy_gd / self.gd_num  / 100000)  # 转化为kj和0.7*100 
        print('加权地面设备平均能耗：' , total_energy_gd / self.gd_num / success_rate  / 100000)
        # print('地面设备平均能耗：', total_energy_gd / total_num_energy * self.gd_num)

        for i in range(len(self.sfns)):
            x = [i for i in range(len(self.cloud.workload_log[i]))]
            plt.plot(x, self.cloud.workload_log[i])
            # 设置标题，显示边缘服务器的索引
            plt.title(f'Workload of Edge Server {i}')
            # 设置x轴和y轴标签
            plt.xlabel('Time Steps')
            plt.ylabel('Workload')
            plt.show()
            np.save('./workload/workload'+str(i)+'.npy', np.array(self.cloud.workload_log[i]))


# def plot(self, episode):
    #     self.comet_ml.log_metric('Value_Loss', self.plotting_dict["value_loss"].item(), episode)
    #     self.comet_ml.log_metric('Grad_Norm_Value', self.plotting_dict["grad_norm_value"], episode)
    #     self.comet_ml.log_metric('Policy_Loss', self.plotting_dict["policy_loss"].item(), episode)
    #     self.comet_ml.log_metric('Grad_Norm_Policy', self.plotting_dict["grad_norm_policy"], episode)
    #     self.comet_ml.log_metric('Entropy', self.plotting_dict["entropy"].item(), episode)
    #     self.comet_ml.log_metric('Threshold_pred', self.plotting_dict["threshold"], episode)
    #
    #     if "threshold" in self.experiment_type:
    #         for i in range(self.num_agents):
    #             agent_name = "agent" + str(i)
    #             self.comet_ml.log_metric('Group_Size_' + agent_name,
    #                                      self.plotting_dict["agent_groups_over_episode"][i].item(), episode)
    #
    #         self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(),
    #                                  episode)
    #
    #         self.comet_ml.log_metric('Num_relevant_agents_in_relevant_set',
    #                                  torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]), episode)
    #         self.comet_ml.log_metric('Num_non_relevant_agents_in_relevant_set',
    #                                  torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]), episode)
    #         self.num_relevant_agents_in_relevant_set.append(
    #             torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]).item())
    #         self.num_non_relevant_agents_in_relevant_set.append(
    #             torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item())
    #         # FPR = FP / (FP+TN)
    #         FP = torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item() * self.num_agents
    #         TN = torch.mean(self.plotting_dict["true_negatives"]).item() * self.num_agents
    #         self.false_positive_rate.append(FP / (FP + TN))
    #
    #     # ENTROPY OF WEIGHTS
    #     entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"] * torch.log(
    #         torch.clamp(self.plotting_dict["weights_value"], 1e-10, 1.0)), dim=2))
    #     self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)