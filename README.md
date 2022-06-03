## Olympics Running Competition

Modified from [https://github.com/jidiai/Competition_Olympics-Running](https://github.com/jidiai/Competition_Olympics-Running).
- clean some code used for Jidi competitions
- new pipeline for developing

### Usage

Training
```shell

# training ppo with random opponent 训练ppo与random的实验
python rl_trainer/main_random.py --device cuda --shuffle_map

# training ppo with ppo 训练ppo与ppo的实验
python rl_trainer/main.py --device cuda --shuffle_map

# training ppo with sacrificing ppo 训练ppo牺牲模型的实验
python rl_trainer/main_sacrifice.py --device cuda --shuffle_map

# training dqn with random opponent 训练dqn与random的实验
python rl_trainer/main_dqn_random.py --device cuda --shuffle_map --algo dqn

# training dqn with dqn 训练dqn与dqn的实验
python rl_trainer/main_dqn.py --device cuda --shuffle_map --algo dqn

# training dqn with sacrificing ppo 训练dqn牺牲模型的实验
python rl_trainer/main_dqn_sacrifice.py --device cuda --shuffle_map --algo dqn
```

Evaluating
```shell
# evaluating ppo with random opponent 模拟ppo与random的比赛
python evaluation.py --my_ai ppo --my_ai_run_dir run5 --map 1

# evaluating ppo with 模拟ppo与ppo的比赛
python evaluation.py --my_ai ppo --my_ai_run_dir run15 --opponent ppo --opponent_run_dir o_run15 --map 1

# evaluating ppo with 模拟ppo牺牲模型与random的比赛
python evaluation.py --my_ai ppo --my_ai_run_dir run22 --map 1

# evaluating ppo with 模拟ppo牺牲模型的比赛
python evaluation.py --my_ai ppo --my_ai_run_dir run22 --opponent ppo_uns --opponent_run_dir o_run22 --map 1

# evaluating dqn with random opponent 模拟dqn与random的比赛
python evaluation.py --my_ai dqn --my_ai_run_dir run1 --map 1

# evaluating dqn with 模拟dqn与dqn的比赛
python evaluation.py --my_ai dqn --my_ai_run_dir run1 --opponent dqn --opponent_run_dir o_run1 --map 1
```
