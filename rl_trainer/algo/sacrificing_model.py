from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.ppo_unsacrifice import PPO_UNS
from rl_trainer.algo.dqn import DQN
from rl_trainer.algo.dqn_unsacrifice import DQN_UNS

from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

class PPO_sacrifice:
    def __init__(
        self, 
        device: str = "cpu",
        run_dir: str = None,
        writer: SummaryWriter = None,
        run_dir_o: str = None,
        writer_o: SummaryWriter = None,
    ):
        self.model = PPO_UNS(device, run_dir, writer)
        self.model_o = PPO(device, run_dir_o, writer_o)
    
    def sacrifice(self):
        self.model_o.actor_net.load_state_dict(self.model.actor_net.state_dict())
        self.model_o.critic_net.load_state_dict(self.model.critic_net.state_dict())
        self.model_o.buffer = deepcopy(self.model.buffer)


class DQN_sacrifice:
    def __init__(
        self, 
        device: str = "cpu",
        run_dir: str = None,
        writer: SummaryWriter = None,
        run_dir_o: str = None,
        writer_o: SummaryWriter = None,
    ):
        self.model = DQN_UNS(device, run_dir, writer)
        self.model_o = DQN(device, run_dir_o, writer_o)
    
    def sacrifice(self):
        self.model_o.q_net.load_state_dict(self.model.q_net.state_dict())
        self.model_o.target_q_net.load_state_dict(self.model.target_q_net.state_dict())
        self.model_o.buffer = deepcopy(self.model.buffer)