import random

import numpy as np
import torch
from torch import nn
import config
from utils import default_device
import torch.nn.functional as F
import math

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        for p in self.gru.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    # x: batch_size, seq
    def forward(self, x):
        out, _ = self.gru(x)
        # 只取最后一个状态
        out = self.out(out[-1:,:,:])
        return out


# MAPPO部分
class RolloutBuffer:
    def __init__(self, capacity, agent_num, state_num, action_num):
        self.states = np.zeros([capacity, agent_num, state_num])
        self.probs = np.zeros([capacity, agent_num, action_num])
        self.logprobs = np.zeros([capacity, agent_num])
        self.actions = np.zeros([capacity, agent_num])
        self.one_hot_actions = np.zeros([capacity, agent_num, action_num])
        # self.probs = []
        # self.logprobs = []
        # self.actions = []
        # self.one_hot_actions = []
        # self.agent_ids = []

        self.rewards = np.zeros([capacity, agent_num])
        self.dones = np.zeros([capacity, agent_num])

        self.values = []
        self.qvalues = []
        # self.agent_global_positions = []

    def clear(self):
        self.actions.fill(0)
        self.probs.fill(0)
        self.logprobs.fill(0)
        self.one_hot_actions.fill(0)
        self.rewards.fill(0)  # 将所有元素设置为 0，保持原形状
        self.dones.fill(0)  # 将所有元素设置为 0，保持原形状

        # 其他 np 数组的清空方式
        self.states.fill(0)  # 将所有元素设置为 0，保持原形状
        self.values.clear()
        self.qvalues.clear()


class Policy(nn.Module):
    def __init__(self, obs_input_dim, num_actions, num_agents, device):
        super(Policy, self).__init__()

        self.name = "MLP Policy"

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.device = device
        self.Policy_MLP = nn.Sequential(
            nn.Linear(obs_input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
            )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

        nn.init.orthogonal_(self.Policy_MLP[0].weight, gain=gain)
        nn.init.orthogonal_(self.Policy_MLP[2].weight, gain=gain)
        nn.init.orthogonal_(self.Policy_MLP[4].weight, gain=gain_last_layer)


    def forward(self, local_observations):
        """
               支持输入为单智能体或者多个智能体的观测。
               - 单智能体输入: [obs_input_dim] -> [num_actions]
               - 多智能体输入: [num_agents, obs_input_dim] -> [num_agents, num_actions]
               """
        # 检查输入维度
        if local_observations.dim() == 1:
            # 输入为单个智能体的观测
            local_observations = local_observations.unsqueeze(0).repeat(self.num_agents, 1)  # 增加批次维度
            action_probs = self.Policy_MLP(local_observations)  # 计算动作概率
            action_probs = action_probs.squeeze(0)  # 去掉批次维度，恢复为 [num_actions]
        else:
            # 输入为多个智能体的观测
            action_probs = self.Policy_MLP(local_observations)  # 计算多个智能体的动作概率

        return action_probs


# using Q network of MAAC
class Q_network(nn.Module):
    def __init__(self, obs_input_dim, num_agents, num_actions, attention_type, device):
        super(Q_network, self).__init__()

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.device = device
        self.attention_type = attention_type

        obs_output_dim = 256
        obs_act_input_dim = obs_input_dim + self.num_actions
        obs_act_output_dim = 256
        curr_agent_output_dim = 128

        self.state_embed = nn.Sequential(
            nn.Linear(obs_input_dim, 256, bias=True),
            nn.Tanh()
        )
        self.key = nn.Linear(256, obs_output_dim, bias=True)
        self.query = nn.Linear(256, obs_output_dim, bias=True)
        if "semi-hard" in self.attention_type:
            self.hard_attention = nn.Sequential(
                nn.Linear(obs_output_dim * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 2)
            )

        self.state_act_embed = nn.Sequential(
            nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=True),
            nn.Tanh()
        )
        self.attention_value = nn.Sequential(
            nn.Linear(obs_act_output_dim, 256, bias=True),
            nn.Tanh()
        )

        self.curr_agent_state_embed = nn.Sequential(
            nn.Linear(obs_input_dim, curr_agent_output_dim, bias=True),
            nn.Tanh()
        )

        # dimesion of key
        self.d_k = obs_output_dim

        # ********************************************************************************************************

        # ********************************************************************************************************
        final_input_dim = obs_act_output_dim + curr_agent_output_dim
        # FCN FINAL LAYER TO GET VALUES
        self.final_value_layers = nn.Sequential(
            nn.Linear(final_input_dim, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, self.num_actions, bias=True)
        )

        # ********************************************************************************************************
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh')

        nn.init.orthogonal_(self.state_embed[0].weight, gain=gain)
        nn.init.orthogonal_(self.state_act_embed[0].weight, gain=gain)

        nn.init.orthogonal_(self.key.weight)
        nn.init.orthogonal_(self.query.weight)
        nn.init.orthogonal_(self.attention_value[0].weight)

        if "semi-hard" in self.attention_type:
            nn.init.orthogonal_(self.hard_attention[0].weight, gain=gain)
            nn.init.orthogonal_(self.hard_attention[2].weight, gain=gain)

        nn.init.orthogonal_(self.curr_agent_state_embed[0].weight, gain=gain)

        nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain)
        nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain)

    def remove_self_loops(self, states_key):
        ret_states_keys = torch.zeros(states_key.shape[0], self.num_agents, self.num_agents - 1, states_key.shape[-1])
        for i in range(self.num_agents):
            if i == 0:
                red_state = states_key[:, i, i + 1:]
            elif i == self.num_agents - 1:
                red_state = states_key[:, i, :i]
            else:
                red_state = torch.cat([states_key[:, i, :i], states_key[:, i, i + 1:]], dim=-2)

            ret_states_keys[:, i] = red_state

        return ret_states_keys.to(self.device)

    def weight_assignment(self, weights):
        weights_new = torch.zeros(weights.shape[0], self.num_agents, self.num_agents).to(self.device)
        one = torch.ones(weights.shape[0], 1).to(self.device)
        for i in range(self.num_agents):
            if i == 0:
                weight_vec = torch.cat([one, weights[:, i, :]], dim=-1)
            elif i == self.num_agents - 1:
                weight_vec = torch.cat([weights[:, i, :], one], dim=-1)
            else:
                weight_vec = torch.cat([weights[:, i, :i], one, weights[:, i, i:]], dim=-1)

            weights_new[:, i] = weight_vec

        return weights_new

    def forward(self, states, policies, actions):
        states_query = states.unsqueeze(-2)
        states_key = states.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        actions_ = actions.unsqueeze(1).repeat(1, self.num_agents, 1, 1)

        states_key = self.remove_self_loops(states_key)
        actions_ = self.remove_self_loops(actions_)

        obs_actions = torch.cat([states_key, actions_], dim=-1)

        # EMBED STATES QUERY
        states_query_embed = self.state_embed(states_query)
        # EMBED STATES QUERY
        states_key_embed = self.state_embed(states_key)
        # KEYS
        key_obs = self.key(states_key_embed)
        # QUERIES
        query_obs = self.query(states_query_embed)
        # WEIGHT
        if 'semi-hard' in self.attention_type:
            query_vector = torch.cat([query_obs.repeat(1, 1, self.num_agents - 1, 1), key_obs], dim=-1)
            hard_weights = nn.functional.gumbel_softmax(self.hard_attention(query_vector), hard=True, dim=-1)
            prop = hard_weights[:, :, :, 1]
            hard_score = -10000 * (1 - prop) + prop
            score = (torch.matmul(query_obs, key_obs.transpose(2, 3)) / math.sqrt(self.d_k)) + hard_score.unsqueeze(-2)
            weight = F.softmax(score, dim=-1)
            weights = self.weight_assignment(weight.squeeze(-2))
        else:
            weight = F.softmax(torch.matmul(query_obs, key_obs.transpose(2, 3)) / math.sqrt(self.d_k), dim=-1)
            weights = self.weight_assignment(weight.squeeze(-2))

        # EMBED STATE ACTION POLICY
        obs_actions_embed = self.state_act_embed(obs_actions)
        attention_values = self.attention_value(obs_actions_embed)
        node_features = torch.matmul(weight, attention_values)

        curr_agent_state_embed = self.curr_agent_state_embed(states)
        curr_agent_node_features = torch.cat([curr_agent_state_embed, node_features.squeeze(-2)], dim=-1)

        Q_value = self.final_value_layers(curr_agent_node_features)

        Value = torch.matmul(Q_value, policies.transpose(1, 2))

        Q_value = torch.sum(actions * Q_value, dim=-1).unsqueeze(-1)

        return Value, Q_value, weights