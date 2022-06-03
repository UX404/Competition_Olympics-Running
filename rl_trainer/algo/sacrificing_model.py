from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.ppo_unsacrifice import PPO_UNS
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