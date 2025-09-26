import random

import numpy as np
from torch import optim

import config
import model
import torch

from torch.distributions import Categorical
# from comet_ml import Experiment

class SFN:
    def __init__(self, id, pos, mode, policy_network, critic_network, policy_network_old, critic_network_old, cloud, dictionary, buffer, sfns=None):
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
        self.n_state = 9
        self.n_action = 5
        self.n_agent = 4
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
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.policy_network_old = policy_network_old
        self.critic_network_old = critic_network_old
        self.device = "cpu"
        self.dictionary = dictionary
        self.buffer = buffer

        self.compare_network = model.Policy(obs_input_dim=9, num_agents=4,num_actions=5, device='cpu').to(self.device)

    def run(self, t):
        if t < 8000:
            self.policy_network.train()
            self.policy_network_old.train()
            self.critic_network.train()
            self.critic_network_old.train()
        else:
            self.policy_network.eval()
            self.policy_network_old.eval()
            self.critic_network.eval()
            self.critic_network_old.eval()

        t_workload = self.statistic_workload()
        if t_workload > 0:
            self.cloud.submit_workload(self.id, self.statistic_workload())
        self.compute_engine(t)
        self.send_to_cloud(t)


        if t > 0 and t % config.MEMORY_CAPACITY_AC == 0:
            # self.update()    # 把全局的网络给复制下来
            #     self.compare_network.load_state_dict(self.policy_network.state_dict())
            self.replay_buffer = np.zeros([config.MEMORY_CAPACITY_AC, self.n_state * 2 + 5 + 2 * self.n_action])     # 把onehot dist等都存进来
        # if t > 0 and t % (config.MEMORY_CAPACITY_AC+1) == 0:
        #     self.compare_model_params(self.compare_network, self.policy_network)


    def statistic_workload(self):
        total_workload = 0
        for i in range(len(self.compute_queue)):
            total_workload += self.compute_queue[i][0].required_res
        return total_workload


    def scheduler(self, t, task, mfn_name):
        if self.mode == 'A3C' or self.mode == 'PRDMAPPO':
            workload = self.cloud.pre_workload(t)
        else:
            workload = [0, 0, 0, 0]
        state = [t - task.start_time, task.result_size, task.task_size, task.required_res, task.deadline, workload]
        action = self.offload(np.hstack(state), t)
        # 这里的state和action都要存一下。关键是每个时间步他都有任务分配吗？
        # self.critic_state.append([self.id, t - task.start_time, task.result_size, task.task_size, task.required_res, task.deadline, workload])
        # self.critic_action.append([self.id, action])
        # self.buffer.states.append(state)
        # self.buffer.agent_ids(self.id)
        if self.index != 0:
            self.replay_buffer[(self.index - 1) % config.MEMORY_CAPACITY_AC][3+self.n_state:3+self.n_state*2] = np.hstack(state)   # 倒数第self.n_state+1 列开始，一直到倒数第2列结束
        tmp = task.name[task.name.find('_')+1:]
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][0] = float(tmp[tmp.find('_')+1:])      # 任务名称
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][1:self.n_state+1] = np.hstack(state)     # 状态信息
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][self.n_state+1] = action               # 以及动作
        self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][-1] = 0
        self.index += 1
        if action == 4:  #len(self.sfns):    # 表示任务被发送到了云端, 0-3分别代表4个边缘服务器。
            self.task_cloud += 1
            send_time = task.task_size / config.core_band + task.task_size / config.ceil_band
            self.send_cloud_queue.append([task, send_time])
        else:
            send_to_sfn = self.sfns[action]    # 发给第几个边缘服务器
            send_to_sfn.compute_queue.append([task, self.id, mfn_name, task.required_res / self.f])

    def compute_engine(self, t):
        total_time = 1
        total_res = 0
        while total_time > 0:
            if len(self.compute_queue) > 0:
                if self.compute_queue[0][-1] <= 1:   # 表示该任务在这个时间步内完成
                    total_time -= self.compute_queue[0][-1]
                    total_res = self.compute_queue[0][-1] * self.f   # 计算任务所消耗的资源量，并将其添加到总资源消耗中。
                    t_task = self.compute_queue.pop(0)
                    if t_task[2] in self.bind_car:    # 发送到该服务器
                        reward = (t - t_task[0].start_time) + (1 - total_time) + (
                                t_task[0].result_size / config.v2i_band)    # (t - t_task[0].start_time) 之前时间步所做的工作的时间开销
                    else:   # cy分析：发送到云 解决跨域问题
                        reward = (t - t_task[0].start_time) + (1 - total_time) + self.cloud.wait_time('send', t) + \
                                 t_task[0].result_size / config.core_band * 2 + t_task[0].result_size / config.ceil_band
                    if reward <= t_task[0].deadline:
                        self.task_success += 1
                        self.task_res += t_task[0].required_res
                        self.time_counter += reward
                    else:
                        self.task_drop += 1
                        reward = 1000
                    send_sfn = self.sfns[t_task[1]]
                    tmp = t_task[0].name[t_task[0].name.find('_') + 1:]   # 如果找到了下划线，返回该下划线的索引（位置），从 0 开始计数。
                    for k in range(len(send_sfn.replay_buffer)):
                        if send_sfn.replay_buffer[k][0] == float(tmp[tmp.find('_')+1:]):
                            send_sfn.replay_buffer[k][self.n_state+2] = reward   # 11的位置是奖励
                            send_sfn.replay_buffer[k][-1] = 1  # 任务完成
                else:
                    self.compute_queue[0][-1] -= 1
                    break
            else:
                break

        self.workload.append(total_res)

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
                            self.replay_buffer[k][self.n_state + 2] = reward  # 11的位置是奖励
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

            # if t < 6000 and t % config.MEMORY_CAPACITY_AC == 0:
            #     self.actor_critic.update_target()
            action = self.get_action(state)
        elif self.mode == 'A3C':
            if t < 6000 and t % config.MEMORY_CAPACITY_AC == 0:
                self.actor_critic.update_target()
            action = self.actor_critic.choose_action(state)
        elif self.mode == 'random':
            action = random.randint(0, self.n_action-1)
        elif self.mode == 'AC':
            if t < 6000 and t % config.MEMORY_CAPACITY_AC == 0:
                self.actor_critic2.update_target()
            action = self.actor_critic2.choose_action(state)
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
                action = dist.argmin().detach().cpu().item()
            else:
                # 取动作分布的倒数并归一化
                inv_dist = 1.0 / (dist + 1e-6)  # 加1e-6防止除零
                inv_dist /= inv_dist.sum()  # 归一化

                # 使用倒数分布进行采样，优先选择原本概率最小的动作
                action = Categorical(inv_dist).sample().detach().cpu().item()

            # 计算动作的 one-hot 编码
            one_hot_action = np.zeros(self.n_action)
            one_hot_action[action] = 1

            probs = Categorical(dist)
            action_logprob = probs.log_prob(torch.tensor(action).to(self.device)).item()
            # print("dist_shape:", dist.shape)
            # print("buffer.shape",self.replay_buffer.shape)
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2 : 3 + self.n_state * 2 + self.n_action] = dist
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2+self.n_action : 3+self.n_state*2+self.n_action*2] = one_hot_action
            self.replay_buffer[self.index % config.MEMORY_CAPACITY_AC][3+self.n_state*2+self.n_action*2] = action_logprob
            # # 保存动作分布、动作的 log 概率和 one-hot 编码
            # self.buffer.probs[self.index, self.id, :] = dist_probs_numpy
            # self.buffer.logprobs[self.index, self.id, :] = action_logprob_numpy
            # self.buffer.one_hot_actions[self.index, self.id, :] = one_hot_action
            return action
    #


    def compare_model_params(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(param1, param2):
                print("Parameters are different.")
        print("Parameters are the same.")