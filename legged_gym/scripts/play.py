import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from pynput import keyboard
import threading
import numpy as np
import torch

# 全局变量存储速度命令
current_lin_vel_x = 0.0 # 前后
current_lin_vel_y = 0.0 # 左右
current_ang_vel_yaw = 0.0 # 转向
lock = threading.Lock()

def on_press(key):
    """键盘按下事件回调函数"""
    global current_lin_vel_x, current_lin_vel_y, current_ang_vel_yaw
    try:
        with lock:
            # W/S: 前后移动
            if key.char == 'w':
                current_lin_vel_x = 0.8
            elif key.char == 's':
                current_lin_vel_x = -0.5
            
            # A/D: 左右平移 (Side Step)
            elif key.char == 'a':
                current_lin_vel_y = 0.5
            elif key.char == 'd':
                current_lin_vel_y = -0.5
            
            # Q/E: 左右旋转 (Turning) - G1需要这个！
            elif key.char == 'q':
                current_ang_vel_yaw = 0.5
            elif key.char == 'e':
                current_ang_vel_yaw = -0.5
                
    except AttributeError:
        pass

def on_release(key):
    """键盘释放事件回调函数"""
    global current_lin_vel_x, current_lin_vel_y, current_ang_vel_yaw
    try:
        with lock:
            if key.char in ('w', 's'):
                current_lin_vel_x = 0.0
            elif key.char in ('a', 'd'):
                current_lin_vel_y = 0.0
            elif key.char in ('q', 'e'):
                current_ang_vel_yaw = 0.0
    except AttributeError:
        pass

def play(args):
    """主运行函数"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # --- 覆盖参数以适应 Play 模式 ---
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)  # 只开几个环境测试
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 【重要】确保相机开启，否则神经网络输入维度会不匹配
    # 虽然不显示图像，但必须产生数据给 Policy 使用
    env_cfg.sensor.stereo_cam.enable = True 
    env_cfg.env.test = True

    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # 加载策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 导出 JIT (可选)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    
    print("\n" + "="*30)
    print("🎮 G1 键盘控制:")
    print("  W / S : 前进 / 后退")
    print("  A / D : 左右平移")
    print("  Q / E : 左右转向")
    print("  Ctrl+C: 退出程序")
    print("="*30 + "\n")
    
    # 主循环
    try:
        while True:
            # 1. 策略推理
            with torch.no_grad():
                actions = policy(obs.detach())
            
            # 2. 应用键盘指令
            with lock:
                # 假设 commands 顺序: [lin_vel_x, lin_vel_y, ang_vel_yaw]
                env.commands[:, 0] = current_lin_vel_x
                env.commands[:, 1] = current_lin_vel_y
                env.commands[:, 2] = current_ang_vel_yaw
                env.commands[:, 3] = 0.0
            
            # 3. 环境步进
            obs, _, rews, dones, infos = env.step(actions.detach())
            # =================================================================
            # 🎥 [可选功能] 相机视角跟随 / 锁定机器人
            # =================================================================
            if True:  # 想要开启时，将这里取消注释或改为 True
                # 获取第0个机器人的真实位置 (CPU numpy格式)
                # root_states: [num_envs, 13], 前3维是位置 x,y,z
                robot_pos = env.root_states[0, :3].cpu().numpy()
                
                # 相机参数配置
                camera_distance = 3.0  # 离机器人多远 (米)
                camera_height = 1.5    # 离地多高 (米)
                
                # 计算相机位置：始终保持在机器人后方一定距离
                # 注意：这里简单假设往Y轴负方向退，如果要更高级的跟随（跟随转向），
                # 需要结合 root_states[0, 3:7] 的四元数来计算后方向量
                camera_pos = np.array([
                    robot_pos[0],                   # X跟随
                    robot_pos[1] - camera_distance, # Y后退
                    robot_pos[2] + camera_height    # Z抬高
                ])
                
                # 计算焦点位置：始终看向机器人中心
                look_at = np.array([
                    robot_pos[0],
                    robot_pos[1],
                    robot_pos[2]
                ])
                
                # 更新 Isaac Gym 的相机视角
                # (前提：你的 env 类封装了 set_camera 方法)
                # if hasattr(env, 'set_camera'):
                #     env.set_camera(camera_pos, look_at)
            # =================================================================
            ######################################
            # base_h = env.root_states[0, 2].item()
            
            # # 这里的 base_vel_z 是基座垂直速度 (索引 7, 8, 9 是线速度 x,y,z)
            # base_vel_z = env.root_states[0, 9].item()

            # # 使用 \r 实现单行刷新，方便观察数值跳变
            # print(f"\r[状态监控] 高度(Z): {base_h:.4f} m | 垂直速度: {base_vel_z:.4f} m/s", end="")

            # 移除了 OpenCV 显示代码
                
    except KeyboardInterrupt:
        print("\n检测到退出信号，程序结束。")
        pass
    
    print("程序结束")


if __name__ == '__main__':
    EXPORT_POLICY = True
    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    args = get_args()
    play(args)