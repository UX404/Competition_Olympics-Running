import os
import sys
from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from rl_trainer.algo.network import Actor, CNN_Actor, CNN_Critic, Critic
from torch.utils.tensorboard import SummaryWriter


class Args:
    gae_lambda = 0.95
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
    buffer_capacity = 1000
    batch_size = 32
    gamma = 0.99
    lr = 0.0001
    epsilon = 0.01
    target_update = 10
    action_space = 36
    state_space = 625


args = Args()

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN_UNS:
    ''' DQN算法 '''
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    ppo_update_time = args.ppo_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    gamma = args.gamma
    action_space = args.action_space
    state_space = args.state_space
    lr = args.lr
    gae_lambda = args.gae_lambda
    use_cnn = False
    epsilon=args.epsilon
    target_update=args.target_update

    def __init__(self,        
        device: str = "cuda",
        run_dir: str = None,
        writer: SummaryWriter = None,
        use_gae: bool = True, ):
        state_dim=self.state_space
        hidden_dim=128
        action_dim=self.action_space 
        learning_rate=self.lr
        gamma=self.gamma
        epsilon=self.epsilon
        target_update=self.target_update
        self.action_dim = action_dim
        self.q_net =  Actor(self.state_space, self.action_space).to(device) # Q网络
        # 目标网络
        self.target_q_net = Actor(self.state_space, self.action_space).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

        self.buffer = []





    def select_action(self, state, train=True):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state.reshape(1,-1), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action,2

    def choose_action(self, state, train=False):
        return self.select_action(state, train)[0]

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.cpu().item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update_target(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    

    def update(self, ep_i, opponent_q_values, opponent_q_targets):
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor([t.action for t in self.buffer], dtype=torch.long)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = [t.reward for t in self.buffer]

        next_states=torch.tensor([t.next_state for t in self.buffer],dtype=torch.float).to(self.device)
        
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.actor_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_actor_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values - opponent_q_values.detach(), q_targets - opponent_q_targets.detach()))
        self.optimizer.zero_grad() 
        dqn_loss.backward()
        self.optimizer.step() 
        if self.count % self.target_update == 0:
            self.target_actor_net.load_state_dict(
                self.actor_net.state_dict())  # 更新目标网络
        self.count += 1
        self.training_step += 1
        self.clear_buffer()
        return q_values, q_targets

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "q_" + str(episode) + ".pth")
        torch.save(self.q_net.state_dict(), model_actor_path)

    def load(self, run_dir, episode):
        print(f"\nBegin to load model: ")
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, "models/olympics-running/dqn")
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, "trained_model")
        model_path = os.path.join(run_path, "q_" + str(episode) + ".pth")
        print(f"q path: {model_path}")

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            q_net = torch.load(model_path, map_location=self.device)
            self.q_net.load_state_dict(q_net)
            print("Model loaded!")
        else:
            sys.exit(f"Model not founded!")
