import numpy as np
import os
import sys
from utils import Task  # 确保从 utils.py 导入 Task 类

# def test_load_tasks():
#     try:
#         # 加载任务文件
#         tasks = np.load('task/car_1_task.npy', allow_pickle=True)
#         print(f"成功加载任务文件")
#         print(f"总时间步数: {len(tasks)}")
        
#         # 检查几个时间步的任务
#         for t in range(min(5, len(tasks))):
#             time_step_tasks = tasks[t]
#             print(f"\n时间步 {t} 的任务数量: {len(time_step_tasks)}")
            
#             # 打印每个时间步的前3个任务的详细信息
#             for i, task in enumerate(time_step_tasks[:3]):
#                 print(f"\n任务 {i+1}:")
#                 print(f"- 名称: {task.name}")
#                 print(f"- 任务大小: {task.task_size}")
#                 print(f"- 结果大小: {task.result_size}")
#                 print(f"- 截止时间: {task.deadline}")
#                 print(f"- 所需资源: {task.required_res}")
#                 print(f"- 开始时间: {task.start_time}")
                
#     except Exception as e:
#         print(f"错误: {str(e)}")
#         print(f"错误类型: {type(e)}")
        
# def test_create_tasks():
#     """测试任务创建"""
#     from utils import create_save_task
    
#     try:
#         # 创建测试任务
#         name = "car_1"
#         index = 0
#         time_steps = 100  # 测试用较小的时间步
        
#         print(f"开始创建测试任务...")
#         final_index = create_save_task(name, index, time_steps)
#         print(f"任务创建完成，最终索引: {final_index}")
        
#         # 加载并验证创建的任务
#         tasks = np.load(f'task/res/22-26/{name}_task.npy', allow_pickle=True)
#         print(f"\n成功加载新创建的任务")
#         print(f"时间步数: {len(tasks)}")
        
#         # 统计信息
#         total_tasks = 0
#         task_sizes = []
#         deadlines = []
        
#         for t in range(min(5, len(tasks))):
#             time_step_tasks = tasks[t]
#             total_tasks += len(time_step_tasks)
            
#             print(f"\n时间步 {t}:")
#             print(f"任务数量: {len(time_step_tasks)}")
            
#             for task in time_step_tasks[:2]:  # 只显示前两个任务
#                 task_sizes.append(task.task_size)
#                 deadlines.append(task.deadline)
#                 print(f"\n- 任务名称: {task.name}")
#                 print(f"  任务大小: {task.task_size}")
#                 print(f"  截止时间: {task.deadline}")
                
#         print(f"\n统计信息:")
#         print(f"检查的总任务数: {total_tasks}")
#         if task_sizes:
#             print(f"任务大小范围: {min(task_sizes)} - {max(task_sizes)}")
#         if deadlines:
#             print(f"截止时间范围: {min(deadlines)} - {max(deadlines)}")
            
#     except Exception as e:
#         print(f"创建任务时出错: {str(e)}")
#         print(f"错误类型: {type(e)}")

# if __name__ == "__main__":
#     print("=== 测试任务加载 ===")
#     test_load_tasks()
#     print("\n=== 测试任务创建 ===")
#     test_create_tasks()
    

data = np.load('route/car_1_route.npy', allow_pickle=True)
print(data.shape)
print(data[:10])