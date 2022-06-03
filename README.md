## Olympics Running Competition

Modified from [https://github.com/jidiai/Competition_Olympics-Running](https://github.com/jidiai/Competition_Olympics-Running).
- clean some code used for Jidi competitions
- new pipeline for developing

### Usage

```shell
git clone https://github.com/Leo-xh/Competition_Olympics-Running.git
cd Competition_Olympics-Running

# training ppo with random opponent
python rl_trainer/main_random.py --device cuda --shuffle_map

# training ppo with ppo
python rl_trainer/main.py --device cuda --shuffle_map

# training ppo with sacrificing ppo
python rl_trainer/main_sacrifice.py --device cuda --shuffle_map

# evaluating ppo with random opponent
python evaluation.py --my_ai ppo --my_ai_run_dir run5 --my_ai_run_episode 800 --map 1

# evaluating ppo with ppo
python evaluation.py --my_ai ppo --my_ai_run_dir run15 --opponent ppo --opponent_run_dir o_run15 --map 1
```

### Suggestions

1. The random opponent may be too weak for developing new algorithms, you can implement other rule-based agents to compete with your algorithm.
2. You can also consider self-paly based training methods in training your agent.
3. For training a ppo algorithm, the given metrics may not be enough, you can add other metrics, e.g. clipping ratio, to help monitoring the training process.
4. Single-agent PPO may not work in difficult maps, and you should train your agent with `--shuffle_map` flag finally.