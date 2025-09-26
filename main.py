import config
import system
from utils import Task
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulation Configuration")
    parser.add_argument("--sfn_offload_mode", type=str, default="PRDMAPPO", help="SFN offload mode (default: PRDMAPPO)")
    parser.add_argument("--mfn_offload_mode", type=str, default="PRDMAPPO", help="MFN offload mode (default: PRDMAPPO)")
    parser.add_argument("--task_gen_rate", type=float, default=0.6, help="Task generation rate (default: 0.6)")
    parser.add_argument("--prediction", type=bool, default=True, help="Tra_prediction (default: True)")
    args = parser.parse_args()

    seed_num = 0 # [0, 1, 2, 3, 4]
    extension = "MAPPO_Q"  # [MAPPO_Q, MAA2C_Q, MAPPO_Q_Semi_Hard_Attn, MAA2C_Q_Semi_Hard_Attn]
    test_num = "MecTest"
    env_name = "MEC"
    experiment_type = "prd_above_threshold"  # prd_above_threshold, shared

    dictionary = {
        "prediction": True,
        "iteration": seed_num,
        "update_type": "ppo",
        "attention_type": "semi-hard",  # [soft, semi-hard]
        "grad_clip_critic": 10.0,
        "grad_clip_actor": 10.0,
        "device": "gpu",
        "critic_dir": 'tests/' + test_num + '/models/' + env_name + '_' + experiment_type + '_' + extension + '/critic_networks/',
        "actor_dir": 'tests/' + test_num + '/models/' + env_name + '_' + experiment_type + '_' + extension + '/actor_networks/',
        "gif_dir": 'tests/' + test_num + '/gifs/' + env_name + '_' + experiment_type + '_' + extension + '/',
        "policy_eval_dir": 'tests/' + test_num + '/policy_eval/' + env_name + '_' + experiment_type + '_' + extension + '/',
        "policy_clip": 0.1,   # 0.1
        "value_clip": 0.1,   # 0.1
        "n_epochs": 10,
        "update_ppo_agent": 400,  
        # "env": env_name
        "test_num": test_num,
        "extension": extension,
        "value_lr": 5e-3,   # 5e-3
        "policy_lr": 5e-3,   # 5e-3
        "entropy_pen": 0.4,    # 0.4
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lambda": 0.95,  # 1 --> Monte Carlo; 0 --> TD(1)
        "select_above_threshold": 0.12,
        "gif": False,
        "gif_checkpoint": 1,
        "load_models": False,
        "model_path_value": "",
        "model_path_policy": "",
        "eval_policy": False,
        "save_model": True,
        "save_model_checkpoint": 1000,
        "save_comet_ml_plot": True,
        "learn": True,
        "max_episodes": 20000,
        "max_time_steps": 70,
        "experiment_type": experiment_type,
        # car部分：
        "car_attention_type": "semi-hard",  # [soft, semi-hard]
        "car_grad_clip_critic": 10.0,
        "car_grad_clip_actor": 10.0,
        "car_policy_clip": 0.1,
        "car_value_clip": 0.1,
        "car_update": 400,
        "car_n_epochs":10,
        "car_value_lr": 5e-4,
        "car_policy_lr":5e-4,
        "car_entropy_pen": 0.4,
        "car_gamma": 0.99,
        "car_gae_lambda": 0.95,
        "car_lambda": 0.95,  
        "car_select_above_threshold": 0.12,
        # station部分：
        "station_attention_type": "semi-hard",  # [soft, semi-hard]
        "station_grad_clip_critic": 20.0,
        "station_grad_clip_actor": 20.0,
        "station_policy_clip": 0.1,
        "station_value_clip": 0.1,
        "station_n_epochs": 10,
        "station_value_lr": 5e-3,
        "station_policy_lr": 5e-3,
        "station_entropy_pen": 0.4,
        "station_gamma": 0.99,
        "station_gae_lambda": 0.95,
        "station_lambda": 0.95,  # 1 --> Monte Carlo; 0 --> TD(1)
        "station_select_above_threshold": 0.12,


    }

    print("\n=== Configuration Settings ===")
    print(f"Seed Number: {seed_num}")
    print(f"Extension: {extension}")
    print(f"Test Number: {test_num}")
    print(f"Environment Name: {env_name}")
    print(f"Experiment Type: {experiment_type}")
    print("\n=== Dictionary Settings ===")
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print("\n=== Simulation Start ===")
    for rn in [12]:  # 只会循环一次 ，rn=4
        sys = system.System(n_state=20, n_action=10, n_mfn=rn, sfn_offload_mode=args.sfn_offload_mode, mfn_offload_mode=args.mfn_offload_mode, task_gen_rate=args.task_gen_rate, dictionary=dictionary)
        sys.start(12)  # 方法开始模拟运行12个时间步
        sys.print()

    # Plotly 数据可视化