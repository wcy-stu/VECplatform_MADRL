import torch.optim

import config
import model
import utils
from utils import default_device


# class Cloud:
#     def __init__(self):
#         self.f = config.f_cloud
#         self.task_success = 0
#         self.task_drop = 0
#         self.time_counter = 0
#         self.workload_log = [[] for _ in range(len(utils.base_pos))]
#         self.workload_p = model.GRU(1, 128, 1).to(default_device())
#         self.optimizer = torch.optim.Adam(params=self.workload_p.parameters(), lr=0.01)
#         self.loss_func = torch.nn.L1Loss()
#         self.epochs = 1000
#         self.batch_size = 16
#         self.look_back = 5
#         self.mode = 'P'
#         self.compute_queue = list()
#         self.send_queue = list()
#
#     def run(self, t):
#         self.exec_task(t)
#         self.send_task()
#
#     def exec_task(self, t):
#         total_time = 1
#         while total_time > 0:
#             if len(self.compute_queue) > 0:
#                 if self.compute_queue[0][1] <= 1:
#                     total_time -= self.compute_queue[0][1]
#                     t_task = self.compute_queue.pop(0)
#                     send_time = t_task[0].result_size / config.core_band + t_task[0].result_size / config.ceil_band
#                     self.send_queue.append([t_task[0], send_time, send_time+self.wait_time('send', t)+t])      # cy:计算完要发送到车辆
#                 else:
#                     self.compute_queue[0][1] -= 1
#                     break
#             else:
#                 break
#
#     def send_task(self):
#         total_time = 3
#         while total_time > 0:
#             if len(self.send_queue) > 0:
#                 if self.send_queue[0][1] <= total_time:
#                     total_time -= self.send_queue[0][1]
#                     self.send_queue.pop(0)
#                 else:
#                     self.send_queue[0][1] -= total_time
#                     total_time = 0
#             else:
#                 break
#
#
#     def receive_task(self, task, t):
#         compute_time = task.required_res/ self.f
#         send_time = task.result_size / config.ceil_band + task.result_size / config.core_band
#         total_time = self.wait_time('exec', t) + compute_time + self.wait_time('send', t) + send_time + t - task.start_time
#         # print(self.wait_time('exec', t), self.wait_time('send', t), t, task.start_time, total_time, task.deadline)
#         if total_time <= task.deadline:
#             self.task_success += 1   # cy:包含计算功能
#             self.time_counter += total_time
#             self.compute_queue.append([task, compute_time, self.wait_time('exec', t)+t+compute_time])
#             return total_time
#         else:
#             self.task_drop += 1
#             return 1000
#
#     def wait_time(self, object, t):
#         if object == 'exec':
#             if len(self.compute_queue) == 0:
#                 return 0
#             else:
#                 return self.compute_queue[-1][-1] - t
#         elif object == 'send':
#             if len(self.send_queue) == 0:
#                 return 0
#             else:
#                 return self.send_queue[-1][-1] - t
#
#
#     def pre_workload(self, t):
#         if self.mode == 'NoneP':
#             return [0 for _ in range(len(utils.base_pos))]
#         else:
#             if t < 1500:
#                 if len(self.workload_log[0]) > 300:
#                     self.workload_p.train()
#                     for i in range(len(self.workload_log)):
#                         # train_sets, target_sets = utils.sample_workload(self.workload_log[i][-100:], self.batch_size, self.look_back)
#                         train_sets, target_sets = utils.sample_workload(self.workload_log[i], self.batch_size, self.look_back)
#                         self.optimizer.zero_grad()
#                         pre_load =self.workload_p(torch.tensor(train_sets, dtype=torch.float).view(self.look_back, self.batch_size, 1).to(default_device())).view(self.batch_size)
#                         target_sets = torch.tensor(target_sets).to(default_device())
#                         loss = self.loss_func(pre_load, target_sets)
#                         loss.backward()
#                         self.optimizer.step()
#                 return [0 for _ in range(len(utils.base_pos))]
#             else:
#                 workload = list()
#                 self.workload_p.eval()
#                 with(torch.no_grad()):  # 确保不会进行梯度计算
#                     for i in range(len(self.workload_log)):
#                         look_back = torch.tensor(self.workload_log[i][-self.look_back:], dtype=torch.float).view(self.look_back, 1, 1).to(default_device())
#                         next_workload = self.workload_p(look_back)
#                         workload.append(next_workload.item())
#                 return workload
#
#     def submit_workload(self, sfn_id, workload):
#         self.workload_log[sfn_id].append(workload)


class HAP:
    def __init__(self):
        self.f = config.f_hap
        self.task_success = 0
        self.task_drop = 0
        self.time_counter = 0
        self.workload_log = [[] for _ in range(len(utils.base_pos))]
        self.workload_p = model.GRU(1, 128, 1).to(default_device())
        self.optimizer = torch.optim.Adam(params=self.workload_p.parameters(), lr=0.01)
        self.loss_func = torch.nn.L1Loss()
        self.epochs = 1000
        self.batch_size = 64  # 增大批量大小，充分利用GPU
        self.look_back = 5
        self.mode = 'NoneP'
        self.compute_queue = list()
        self.send_queue = list()

    def run(self, t):
        self.exec_task(t)
        self.send_task()

    def exec_task(self, t):
        total_time = 1
        while total_time > 0:
            if len(self.compute_queue) > 0:
                if self.compute_queue[0][1] <= 1:
                    total_time -= self.compute_queue[0][1]
                    t_task = self.compute_queue.pop(0)
                    send_time = t_task[0].result_size / config.g2h_band
                    self.send_queue.append([t_task[0], send_time, send_time + self.wait_time('send', t) + t])
                else:
                    self.compute_queue[0][1] -= 1
                    break
            else:
                break

    def send_task(self):
        total_time = 3
        while total_time > 0:
            if len(self.send_queue) > 0:
                if self.send_queue[0][1] <= total_time:
                    total_time -= self.send_queue[0][1]
                    self.send_queue.pop(0)
                else:
                    self.send_queue[0][1] -= total_time
                    total_time = 0
            else:
                break

    def receive_task(self, task, t):
        compute_time = task.required_res / self.f
        send_time = task.result_size / config.g2h_band
        total_time = self.wait_time('exec', t) + compute_time + self.wait_time('send',
                                                                               t) + send_time + t - task.start_time

        if total_time <= task.deadline:
            self.task_success += 1
            self.time_counter += total_time
            self.compute_queue.append([task, compute_time, self.wait_time('exec', t) + t + compute_time])
            return total_time
        else:
            self.task_drop += 1
            return 1000

    def wait_time(self, object, t):
        if object == 'exec':
            return self.compute_queue[-1][-1] - t if self.compute_queue else 0
        elif object == 'send':
            return self.send_queue[-1][-1] - t if self.send_queue else 0

    def pre_workload(self, t):
        if self.mode == 'NoneP':
            # return [0 for _ in range(len(utils.base_pos))]
            return self.get_current_workloads()
        else:
            if t < 1500:
                if len(self.workload_log[0]) > 300:
                    self.workload_p.train()
                    for i in range(len(self.workload_log)):
                        train_sets, target_sets = utils.sample_workload(self.workload_log[i], self.batch_size,
                                                                        self.look_back)

                        # 确保数据在GPU上处理
                        train_sets = torch.tensor(train_sets, dtype=torch.float).view(self.look_back, self.batch_size,
                                                                                      1).to(default_device())
                        target_sets = torch.tensor(target_sets, dtype=torch.float).to(default_device())

                        self.optimizer.zero_grad()
                        pre_load = self.workload_p(train_sets).view(self.batch_size)
                        loss = self.loss_func(pre_load, target_sets)
                        loss.backward()
                        self.optimizer.step()
                return [0 for _ in range(len(utils.base_pos))]
            else:
                workload = list()
                self.workload_p.eval()
                with torch.no_grad():
                    look_back_data = [
                        torch.tensor(self.workload_log[i][-self.look_back:], dtype=torch.float).view(self.look_back, 1,
                                                                                                     1).to(
                            default_device())
                        for i in range(len(self.workload_log))]

                    # 批量处理预测，减少GPU-CPU同步操作
                    next_workloads = [self.workload_p(look_back).item() for look_back in look_back_data]
                    workload.extend(next_workloads)
                return workload

    def submit_workload(self, sfn_id, workload):
        self.workload_log[sfn_id].append(workload)

    def get_current_workloads(self):
        current_loads = []
        for i in range(len(self.workload_log)):
            if len(self.workload_log[i]) > 0:
                # 获取当前智能体的最新负载（最后一个元素）
                current_loads.append(self.workload_log[i][-1])
            else:
                # 如果该智能体还没有负载记录，默认负载为0
                current_loads.append(0)
        return current_loads