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
import matplotlib.pyplot as plt # 显式导入用于绘图

# 全局变量存储速度命令
current_lin_vel_x = 0.0 # 前后
current_lin_vel_y = 0.0 # 左右
current_ang_vel_yaw = 0.0 # 转向
lock = threading.Lock()

def _get_key_char(key):
    try:
        if hasattr(key, 'char') and key.char is not None:
            return key.char.lower()
    except Exception:
        pass
    return None

def on_press(key):
    """键盘按下事件回调函数"""
    global current_lin_vel_x, current_lin_vel_y, current_ang_vel_yaw
    try:
        with lock:
            key_char = _get_key_char(key)
            # W/S: 前后移动
            if key_char == 'w' or key == keyboard.Key.up:
                current_lin_vel_x = 0.8
            elif key_char == 's' or key == keyboard.Key.down:
                current_lin_vel_x = -0.5
            
            # A/D: 左右平移 (Side Step)
            elif key_char == 'a' or key == keyboard.Key.left:
                current_lin_vel_y = 0.5
            elif key_char == 'd' or key == keyboard.Key.right:
                current_lin_vel_y = -0.5
            
            # Q/E: 左右旋转 (Turning) - G1需要这个！
            elif key_char == 'q':
                current_ang_vel_yaw = 0.5
            elif key_char == 'e':
                current_ang_vel_yaw = -0.5
                
    except AttributeError:
        pass

def on_release(key):
    """键盘释放事件回调函数"""
    global current_lin_vel_x, current_lin_vel_y, current_ang_vel_yaw
    try:
        with lock:
            key_char = _get_key_char(key)
            if key_char in ('w', 's') or key in (keyboard.Key.up, keyboard.Key.down):
                current_lin_vel_x = 0.0
            elif key_char in ('a', 'd') or key in (keyboard.Key.left, keyboard.Key.right):
                current_lin_vel_y = 0.0
            elif key_char in ('q', 'e'):
                current_ang_vel_yaw = 0.0
    except AttributeError:
        pass

def play(args):
    """主运行函数"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # --- 覆盖参数以适应 Play 模式 ---
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)  # 只开几个环境测试
    env_cfg.init_state.pos = [0.0, 0.0, 0.8]
    # Play 固定楼梯评测集：0.11~0.15m 递增台阶（5x5，越往后越高）
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
    env_cfg.terrain.stair_height_levels = [0.11, 0.12, 0.13, 0.14, 0.15]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e9
    # Play 手动评测：保留视觉回调，但仅在前进任务时允许其接管 yaw
    env_cfg.env.manual_cmd_override = False
    env_cfg.env.allow_test_resample = False
    
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
    # 设置存储路径
    log_dir = "/home/zzz/unitree_rl_gym/logs/g1/play"
    os.makedirs(log_dir, exist_ok=True)

    # 自定义记录器字典
    data_log = {}
    play_reward_log_keys = getattr(env_cfg.env, "episode_reward_log_keys", None)
    if isinstance(play_reward_log_keys, (list, tuple, set)) and len(play_reward_log_keys) > 0:
        play_reward_log_keys = set(play_reward_log_keys)
    else:
        play_reward_log_keys = None
    robot_index = 0
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
    
    # 计数器，用于控制记录频率（可选）
    step_counter = 0
    record_every_n_steps = 1 # 1表示每一步都记，想要减小数据量可以改大
    print_every_n_steps = 100

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
                # 当前为 heading_command=False，必须写到 commands[:, 2]
                env.commands[:, 2] = current_ang_vel_yaw
                req_cmd_x, req_cmd_y, req_cmd_yaw = current_lin_vel_x, current_lin_vel_y, current_ang_vel_yaw
            
            # 3. 环境步进
            applied_cmd = env.commands[0, :3].detach().cpu().clone()
            obs, _, rews, dones, infos = env.step(actions.detach())
            if step_counter % print_every_n_steps == 0:
                vel_x = env.base_lin_vel[0, 0].item()
                vel_y = env.base_lin_vel[0, 1].item()
                yaw_rate = env.base_ang_vel[0, 2].item()
                if hasattr(env, "rpy"):
                    body_roll = env.rpy[0, 0].item()
                    body_pitch = env.rpy[0, 1].item()
                    body_yaw = env.rpy[0, 2].item()
                else:
                    body_roll = float("nan")
                    body_pitch = float("nan")
                    body_yaw = float("nan")
                exec_cmd_x = env.commands[0, 0].item()
                exec_cmd_y = env.commands[0, 1].item()
                exec_cmd_yaw = env.commands[0, 2].item()
                done0 = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
                print(
                    f"[PLAY CMD] req(x={req_cmd_x:.2f}, y={req_cmd_y:.2f}, yaw={req_cmd_yaw:.2f}) "
                    f"| applied(x={applied_cmd[0]:.2f}, y={applied_cmd[1]:.2f}, yaw={applied_cmd[2]:.2f}) "
                    f"| post(x={exec_cmd_x:.2f}, y={exec_cmd_y:.2f}, yaw={exec_cmd_yaw:.2f}) "
                    f"| done={int(done0)} "
                    f"| vel_x={vel_x:.2f}, vel_y={vel_y:.2f}, yaw_rate={yaw_rate:.2f}, "
                    f"body_roll={body_roll:.2f}, body_pitch={body_pitch:.2f}, body_yaw={body_yaw:.2f}"
                )
            # --- 实时记录逻辑 ---
            # --- 实时记录逻辑 ---
            if step_counter % record_every_n_steps == 0:
                # A. 基础物理状态
                if 'base_height' not in data_log: data_log['base_height'] = []
                if 'vel_x' not in data_log: data_log['vel_x'] = []
                if 'total_reward' not in data_log: data_log['total_reward'] = []
                
                data_log['base_height'].append(env.root_states[robot_index, 2].item())
                data_log['vel_x'].append(env.base_lin_vel[robot_index, 0].item())
                data_log['total_reward'].append(rews[robot_index].item())

                # B. 动态记录所有分项奖励
                for name in env.reward_names:
                    if (play_reward_log_keys is not None) and (name not in play_reward_log_keys):
                        continue
                    key = f'rew_{name}'
                    if key not in data_log: data_log[key] = []
                    # 直接读取环境在本步已计算好的分项奖励，避免二次调用有状态奖励函数污染结果
                    if hasattr(env, 'step_reward_terms') and name in env.step_reward_terms:
                        val = env.step_reward_terms[name][robot_index].detach().cpu().item()
                    else:
                        val = np.nan
                    data_log[key].append(val)

            # 相机跟随逻辑
            if True:
                robot_pos = env.root_states[0, :3].cpu().numpy()
                camera_pos = np.array([robot_pos[0], robot_pos[1] - 3.0, robot_pos[2] + 1.5])
                look_at = np.array([robot_pos[0], robot_pos[1], robot_pos[2]])
                # if hasattr(env, 'set_camera'): env.set_camera(camera_pos, look_at)
            
            step_counter += 1
            # =================================================================
            # 🎥 [可选功能] 相机视角跟随 / 锁定机器人
            # =================================================================
                
    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] 正在停止运行并生成图表...")
    except Exception as e:
        print(f"\n[ERROR] 程序发生异常: {e}")

    finally:
        if data_log and len(data_log.get('total_reward', [])) > 0:
            print(f"正在清理 GPU 数据并保存至 {log_dir}...")
            
            # 【新增】强制清理数据，确保绘图前全是 Numpy 数组
            clean_log = {}
            for k, v in data_log.items():
                clean_log[k] = [x.item() if torch.is_tensor(x) else x for x in v]
            
            np.save(os.path.join(log_dir, 'play_data.npy'), clean_log)
            
            # 绘图使用 clean_log
            num_charts = len(clean_log)
            cols = 3
            rows = (num_charts + cols - 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
            axs = axs.flatten()
            
            for i, (name, values) in enumerate(clean_log.items()):
                axs[i].plot(values)
                axs[i].set_title(name)
                axs[i].grid(alpha=0.3)
                axs[i].set_xlabel('Steps')
            
            # 隐藏多余的空白子图
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'play_curves.png'))
            print(f"🎉 成功！分项奖励曲线已保存至: {log_dir}/play_curves.png")
            # plt.show() 
        else:
            print("未记录到有效数据，无法生成曲线图。")


if __name__ == '__main__':
    EXPORT_POLICY = True
    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    args = get_args()
    play(args)
