import os
import numpy as np
from datetime import datetime
import sys

# 优先使用工程内 rsl_rl 代码，避免落到外部安装路径
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOCAL_RSL_RL = os.path.join(_ROOT_DIR, "rsl_rl-main")
if os.path.isdir(_LOCAL_RSL_RL) and (_LOCAL_RSL_RL not in sys.path):
    sys.path.insert(0, _LOCAL_RSL_RL)

# W&B 进程启动稳定性：默认使用 thread，且给更长服务等待时间
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
