# -*- coding:utf-8  -*-
# Time  : 2021/02/26 16:25
# Author: Yutong Wu

from pathlib import Path
import os
import yaml


def make_logpath(game_name, algo):
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path("models") / game_name / algo
    if not model_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
            curr_run_o = "o_run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
            curr_run_o = "o_run%i" % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir
    run_dir_o = model_dir / curr_run_o
    log_dir_o = run_dir_o
    return run_dir, log_dir, run_dir_o, log_dir_o


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), "config.yaml"), mode="w", encoding="utf-8")
    yaml.dump(vars(args), file)
    file.close()
