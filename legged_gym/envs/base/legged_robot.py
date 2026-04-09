from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 记录本回合“已到达的最远距离”（用于课程升级判定）
        if hasattr(self, "episode_start_xy") and hasattr(self, "episode_max_distance"):
            cur_distance = torch.norm(self.root_states[:, :2] - self.episode_start_xy, dim=1)
            self.episode_max_distance = torch.maximum(self.episode_max_distance, cur_distance)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # ================= [核心补丁] 激活课程学习 =================
        # 1. 更新地形课程：根据表现让机器人去更难的地形
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # 2. 更新命令课程：根据表现增加指令速度上限
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # 记录新回合起点，并清空“本回合最远距离”
        if hasattr(self, "episode_start_xy"):
            self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        if hasattr(self, "episode_max_distance"):
            self.episode_max_distance[env_ids] = 0.0

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # 兜底：防止奖励函数返回 python 标量导致后续 detach 失败
            if not torch.is_tensor(rew):
                rew = torch.full((self.num_envs,), float(rew), dtype=torch.float, device=self.device)
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if hasattr(self, "step_reward_terms"):
                self.step_reward_terms[name] = rew.detach()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            if hasattr(self, "step_reward_terms"):
                self.step_reward_terms["termination"] = rew.detach()
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = float(self.cfg.domain_rand.max_push_vel_xy)

        # 可选：推搡课程化。前期弱/关推搡，后期再逐步拉起，减少对主任务学习的污染。
        if bool(getattr(self.cfg.domain_rand, "push_curriculum", False)):
            push_start_progress = float(getattr(self.cfg.domain_rand, "push_start_progress", 0.45))
            push_min_scale = float(getattr(self.cfg.domain_rand, "push_min_scale", 0.0))
            den = max(1.0 - push_start_progress, 1e-6)
            if hasattr(self, "current_terrain_progress"):
                prog_all = self.current_terrain_progress
                if not torch.is_tensor(prog_all):
                    prog_all = torch.full((self.num_envs,), float(prog_all), device=self.device)
                elif prog_all.ndim == 0:
                    prog_all = prog_all.repeat(self.num_envs)
                else:
                    prog_all = prog_all.to(self.device)
                prog = prog_all[push_env_ids]
            else:
                prog = torch.zeros(len(push_env_ids), device=self.device)

            push_scale = torch.clamp((prog - push_start_progress) / den, min=0.0, max=1.0)
            push_scale = torch.clamp(push_min_scale + (1.0 - push_min_scale) * push_scale,
                                     min=push_min_scale, max=1.0)
            max_vel_env = max_vel * push_scale
            active = max_vel_env > 1e-4
            if not active.any():
                return
            push_env_ids = push_env_ids[active]
            max_vel_env = max_vel_env[active]
            rand_xy = 2.0 * torch.rand((len(push_env_ids), 2), device=self.device) - 1.0
            # 仅对被选中的环境施加推搡，避免污染其余环境的状态缓存
            self.root_states[push_env_ids, 7:9] = rand_xy * max_vel_env.unsqueeze(1)
        else:
            # 仅对被选中的环境施加推搡，避免污染其余环境的状态缓存
            self.root_states[push_env_ids, 7:9] = torch_rand_float(
                -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
            ) # lin vel x/y
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    def _update_terrain_curriculum(self, env_ids):
        """ 
        [修复版] 根据机器人实际行进的相对距离调整难度。
        解决因 init_state.pos 偏移导致的虚假进阶问题。
        """
        if not self.init_done:
            return
        
        # 1. 终点位移（用于降级稳定性判定）
        final_distance = torch.norm(self.root_states[env_ids, :2] - self.episode_start_xy[env_ids], dim=1)
        # 2. 本回合最大位移（用于升级能力判定）
        max_distance = self.episode_max_distance[env_ids]

        if len(final_distance) > 0:
            if not hasattr(self, "extras"):
                self.extras = {}
            # 兼容原有看板：distance 指标改为“本回合最远距离”
            self.extras["Metrics/mean_distance"] = torch.mean(max_distance).item()
            self.extras["Metrics/max_distance"] = torch.max(max_distance).item()
            # 附加终点位移，便于对比“能力”与“收官稳定”
            self.extras["Metrics/mean_distance_final"] = torch.mean(final_distance).item()
            self.extras["Metrics/max_distance_final"] = torch.max(final_distance).item()
        # 3. 升级逻辑：支持按任务配置覆盖升级距离阈值
        move_up_distance = getattr(self.cfg.terrain, "curriculum_move_up_distance",
                                   self.cfg.terrain.terrain_length / 2.0)
        # 抗推搡污染：升级不仅看 max_distance，还要求回合末仍保持足够位移
        # 防止“被外力推远一瞬间”误判为可升级能力。
        move_up_final_ratio = float(getattr(self.cfg.terrain, "curriculum_move_up_final_ratio", 0.45))
        dist_gate = max_distance > move_up_distance
        final_gate = final_distance > move_up_distance * move_up_final_ratio
        move_up = dist_gate & final_gate
        # 任务相关升级门槛：需要有足够楼梯暴露占比，避免“平地跑很远也升级”
        move_up_stair_conf = float(getattr(self.cfg.terrain, "curriculum_move_up_stair_conf", 0.0))
        stair_gate = torch.ones_like(move_up, dtype=torch.bool)
        nav_gate = torch.ones_like(move_up, dtype=torch.bool)
        if move_up_stair_conf > 0.0 and hasattr(self, "episode_stair_conf_sum"):
            if hasattr(self, "episode_nav_steps"):
                nav_steps_raw = self.episode_nav_steps[env_ids].float()
            else:
                nav_steps_raw = self.episode_length_buf[env_ids].float()
            nav_steps = torch.clamp(nav_steps_raw, min=1.0)
            mean_stair_conf = self.episode_stair_conf_sum[env_ids] / nav_steps
            peak_thr = float(getattr(self.cfg.terrain, "curriculum_move_up_stair_conf_peak",
                                     move_up_stair_conf + 0.20))
            nav_steps_min = float(getattr(self.cfg.terrain, "curriculum_move_up_nav_steps_min", 20.0))
            peak_stair_conf = self.episode_stair_conf_max[env_ids] if hasattr(self, "episode_stair_conf_max") \
                else mean_stair_conf
            stair_ok = (mean_stair_conf > move_up_stair_conf) | (peak_stair_conf > peak_thr)
            nav_ok = nav_steps_raw >= nav_steps_min
            stair_gate = stair_ok
            nav_gate = nav_ok
            move_up = move_up & stair_gate & nav_gate

        if len(env_ids) > 0:
            if not hasattr(self, "extras"):
                self.extras = {}
            # 升级诊断指标：显式拆分每个 gate，避免把 stair_gate 误读为最终升级条件
            stair_nav_gate = stair_gate & nav_gate
            joint_gate = dist_gate & stair_gate & nav_gate
            self.extras["Metrics/move_up_dist_gate_rate"] = torch.mean(dist_gate.float()).item()
            self.extras["Metrics/move_up_stair_gate_rate"] = torch.mean(stair_gate.float()).item()
            self.extras["Metrics/move_up_nav_gate_rate"] = torch.mean(nav_gate.float()).item()
            self.extras["Metrics/move_up_stair_nav_gate_rate"] = torch.mean(stair_nav_gate.float()).item()
            self.extras["Metrics/move_up_joint_gate_rate"] = torch.mean(joint_gate.float()).item()
            self.extras["Metrics/move_up_ready_rate"] = torch.mean(move_up.float()).item()

            # 平地转圈倾向诊断：横摆/偏航相对前向推进的比值
            if hasattr(self, "episode_abs_vx_sum") and hasattr(self, "episode_abs_vy_sum") and \
               hasattr(self, "episode_abs_wz_sum") and hasattr(self, "episode_nav_steps"):
                nav_steps_diag = torch.clamp(self.episode_nav_steps[env_ids].float(), min=1.0)
                mean_abs_vx = self.episode_abs_vx_sum[env_ids] / nav_steps_diag
                mean_abs_vy = self.episode_abs_vy_sum[env_ids] / nav_steps_diag
                mean_abs_wz = self.episode_abs_wz_sum[env_ids] / nav_steps_diag
                circle_index = (mean_abs_vy + 0.35 * mean_abs_wz) / (mean_abs_vx + 0.05)
                self.extras["Metrics/mean_circle_index_episode"] = torch.mean(circle_index).item()

        # ==========================================================
        # [新增日志]：打印触发升级的机器人的地形原点和升级位置
        # ==========================================================
        curriculum_verbose = bool(getattr(self.cfg.env, "log_curriculum_events", False))
        if curriculum_verbose and move_up.any():
            # 获取真正触发升级的环境ID (利用 move_up 作为掩码)
            upgraded_env_ids = env_ids[move_up]
            
            # 提取对应的 地形原点 和 当前根节点坐标 (转移到CPU方便打印)
            up_origins = self.env_origins[upgraded_env_ids].cpu().numpy()
            up_current_pos = self.root_states[upgraded_env_ids, :3].cpu().numpy()
            
            print(f"\n[Terrain Curriculum] 触发地形升级! (共 {len(upgraded_env_ids)} 个机器人)")
            # 遍历打印每个升级机器人的详细坐标 (保留两位小数)
            for i in range(len(upgraded_env_ids)):
                e_id = upgraded_env_ids[i].item()
                origin = up_origins[i]
                curr_pos = up_current_pos[i]
                max_dist = max_distance[move_up][i].item()
                print(f"  -> 环境ID: {e_id:04d} | max_distance: {max_dist:5.2f} | 地形原点: [{origin[0]:6.2f}, {origin[1]:6.2f}, {origin[2]:6.2f}] | 升级触发位置: [{curr_pos[0]:6.2f}, {curr_pos[1]:6.2f}, {curr_pos[2]:6.2f}]")
            print("-" * 60)
        # ==========================================================
        # 4. 降级逻辑：
        # 去掉“仅短回合可降级”的硬条件，避免长回合几乎无法降级。
        # 同时加入 max_distance 约束，避免“过程有推进、终点回摆”被误降。
        move_down_distance = float(getattr(self.cfg.terrain, "curriculum_move_down_distance", 1.0))
        move_down_max_ratio = float(getattr(self.cfg.terrain, "curriculum_move_down_max_ratio", 0.55))
        move_down = (final_distance < move_down_distance) & \
                    (max_distance < move_up_distance * move_down_max_ratio) & \
                    (~move_up)

        # [核心优化]：只有在 Level 1 及以上的机器人，才允许真正触发降级！
        # 防止 Level 0 的机器人在新手村无限触发无效降级并疯狂刷屏。
        is_not_level_zero = self.terrain_levels[env_ids] > 0
        valid_move_down = move_down & is_not_level_zero

        # [连续降级保护]：必须连续2次才真正降级，防止随机扰动导致误降
        # 注意：valid_move_down 是对 env_ids 的掩码，需要用 env_ids[valid_move_down] 来索引
        downgrade_env_ids = env_ids[valid_move_down]
        self.consecutive_downgrade_count[downgrade_env_ids] += 1
        reset_env_ids = env_ids[~valid_move_down]
        self.consecutive_downgrade_count[reset_env_ids] = 0  # 重置计数

        # 只有连续2次才触发真正降级
        # 注意：需要用env_ids索引consecutive_downgrade_count，获取当前batch的计数
        count_in_batch = self.consecutive_downgrade_count[env_ids]
        real_downgrade = valid_move_down & (count_in_batch >= 2)

        if curriculum_verbose and real_downgrade.any():
            # 使用过滤后的掩码 real_downgrade
            downgraded_env_ids = env_ids[real_downgrade]

            down_origins = self.env_origins[downgraded_env_ids].cpu().numpy()
            down_current_pos = self.root_states[downgraded_env_ids, :3].cpu().numpy()

            print(f"\n[Terrain Curriculum] 触发地形真·降级! (共 {len(downgraded_env_ids)} 个机器人被打回低难度)")
            for i in range(len(downgraded_env_ids)):
                e_id = downgraded_env_ids[i].item()
                origin = down_origins[i]
                curr_pos = down_current_pos[i]
                cnt = self.consecutive_downgrade_count[downgraded_env_ids[i]].item()
                print(f"  -> 环境ID: {e_id:04d} | 连续降级次数: {cnt} | 地形原点: [{origin[0]:6.2f}, {origin[1]:6.2f}, {origin[2]:6.2f}] | 降级触发位置: [{curr_pos[0]:6.2f}, {curr_pos[1]:6.2f}, {curr_pos[2]:6.2f}]")
            print("-" * 60)

        # 5. 更新等级并限制范围 [0, 9] (注意这里减去的是 real_downgrade)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * real_downgrade
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, self.cfg.terrain.num_rows - 1)
        
        # 6. 更新下一次重置的“环境原点”（分配房间）
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # [新增] 记录每个环境所在的行（等级）
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # [新增] 记录每个环境所在的列（类型：0为平缓，1为粗糙，2为上楼，3为下楼...）
        # 将环境均匀分配到 20 个地形类型中
        num_cols = self.cfg.terrain.num_cols
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                       (self.num_envs / num_cols), rounding_mode='floor').to(torch.long)
        # [新增] 记录连续降级次数，只有连续2次才真正降级
        self.consecutive_downgrade_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # [新增] 回合起点与本回合最远位移（用于折中课程：升级看能力，降级看稳定）
        self.episode_start_xy = self.root_states[:, :2].clone()
        self.episode_max_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        # 每步分项奖励（用于 play/debug，避免二次调用有状态奖励函数）
        self.step_reward_terms = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                  for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        """
        [死区容忍版] 允许正常的上楼/下楼速度，只严惩剧烈的弹跳和失控坠落。
        """
        # 1. 取垂直速度的绝对值
        vel_z_abs = torch.abs(self.base_lin_vel[:, 2])
        
        # 2. 设定 0.2 m/s 的合法死区 (允许机器人正常上下 0.1~0.2m 的台阶)
        # 只有当速度超出 0.2 m/s 时，才开始计算误差
        excess_vel = torch.clamp(vel_z_abs - 0.1, min=0.0)
        
        # 3. 转化为 0~1 的惩罚项 (超出越多，越接近 1.0)
        sigma = 0.1 
        return 1.0 - torch.exp(-torch.square(excess_vel) / sigma)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        [姿态解耦版] 宽容前倾(Pitch)，严惩侧翻(Roll)
        """
        # 0 代表 Pitch，乘以一个小系数(如 0.1)，允许爬楼时适当弯腰/前倾
        pitch_error = torch.square(self.projected_gravity[:, 0]) * 0.5
        
        # 1 代表 Roll，乘以一个大系数(如 2.0)，严厉禁止左右侧歪
        roll_error = torch.square(self.projected_gravity[:, 1]) * 1.5
        
        return pitch_error + roll_error

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        base_reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        # 0 指令附近软门控：抑制“无指令漂移也吃到 tracking 正奖”
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        cmd_gate = torch.sigmoid((cmd_norm - 0.05) / 0.015)
        return base_reward * cmd_gate
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_ang_vel(self):
    #     """
    #     [核心修复 1]：免除视觉对齐时的转弯追踪惩罚
    #     """
    #     # 1. 正常计算角速度误差
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])

    #     # 2. 判断是否在楼梯地形上（用高度极差判断）
    #     if self.cfg.terrain.mesh_type == 'trimesh' and hasattr(self, 'measured_heights'):
    #         h_max = torch.max(self.measured_heights, dim=1)[0]
    #         h_min = torch.min(self.measured_heights, dim=1)[0]
    #         on_stairs = (h_max - h_min > 0.04)

    #         # 👑 魔法豁免权：在台阶上 且 宏观指令要求直走 (|cmd| < 0.01) 时，
    #         # 说明任何发生的转弯都是视觉系统在”救命”。强行将误差清零！
    #         ang_vel_error = torch.where(on_stairs & (torch.abs(self.commands[:, 2]) < 0.01),
    #                                     torch.zeros_like(ang_vel_error),
    #                                     ang_vel_error)

    #     return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_lin_vel(self):
    #     """ [带门控的速度奖励]：只有方向走对了，给速度才给分 """
    #     # 1. 计算前向速度得分
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     vel_reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        
    #     # 2. 计算方向盘得分
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     ang_reward = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        
    #     # 3. 乘法门控：转弯逃跑时，这里直接拿 0 分
    #     return vel_reward * ang_reward
    
    # def _reward_tracking_ang_vel(self):
    #     """ [独立的转向兜底]：保证机器人原地转弯时也有收益 """
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        """
        [终极分段引导版] 阶梯诱导 + 突变跃升 + 目标逼近
        1. 未达阈值时，给予微小线性奖励（搭小梯子，保留梯度）。
        2. 跨越阈值瞬间，奖励突变放大（给大甜头）。
        3. 跨越阈值后，越接近 target_air_time (0.35s)，奖励越平滑逼近 1.0。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # 仅在触地那一瞬间计算奖励
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        
        # --- 1. 计算高斯平滑基础奖励 (终极目标 0.35s) ---
        target_air_time = 0.35
        sigma = 0.1 
        air_time_error = torch.square(self.feet_air_time - target_air_time)
        # 此时 rew_base 是一个以 0.35 为中心，最高为 1.0 的平滑曲线
        rew_base = torch.exp(-air_time_error / sigma) 
        
        # --- 2. 获取动态课程门槛 ---
        progress = getattr(self, 'current_terrain_progress', 0.0)
        if isinstance(progress, torch.Tensor):
            progress = progress.unsqueeze(1)
            
        dynamic_threshold = 0.15 + 0.10 * progress
        
        # --- 3. 👑 核心：分段突变逻辑 ---
        # 【小梯子】：低于阈值时，最多只给 20% 的权重，保留微小梯度让它知道要继续抬
        gradient_scale = 0.2 * (self.feet_air_time / dynamic_threshold)
        
        # 判断是否跨越及格线
        is_above_threshold = self.feet_air_time > dynamic_threshold
        
        # 跨越及格线后，不再压制，权重直接解放为 1.0！
        # 但此时它的实际得分仍由 rew_base 决定，只有逼近 0.35s 才能拿满。
        final_scale = torch.where(is_above_threshold, torch.tensor(1.0, device=self.device), gradient_scale)
        
        # 融合得出最终的单次滞空得分
        rew_airTime = rew_base * final_scale

        # --- 4. 结算与重置 ---
        res = torch.sum(rew_airTime * first_contact, dim=1)
        
        # 只有在有前向/侧向速度指令时，才给滞空奖励，防止原地踏步刷分
        res *= torch.norm(self.commands[:, :2], dim=1) > 0.1 
        
        # 重置刚刚触地的那只脚的计时器
        self.feet_air_time *= ~contact_filt
        
        return res
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # 1. 没指令乱动的惩罚
        no_cmd_motion = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        
        # 2. 有指令但不往前的惩罚 (怠工惩罚)
        # 只要你要求前进速度 > 0.5，但它的实际世界速度 < 0.1，就狠狠扣分！
        has_cmd = (self.commands[:, 0] > 0.5)
        not_moving = (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.1)
        laziness = (has_cmd & not_moving).float() * 1.0  # 怠工常数惩罚
        
        return no_cmd_motion + laziness
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
