from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import warp as wp
import sys
import os
# 调试 CUDA 内核错误时可临时设置：CUDA_LAUNCH_BLOCKING=1
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sensors.warp.warp_stereo_cam import WarpStereoCam
from legged_gym.utils.terrain import Terrain  # 导入机器人配置
from legged_gym.utils.math import wrap_to_pi
import math
from typing import Tuple
# ✅ 添加Warp导入和初始化
import numpy as np
import cv2
import pandas as pd
from sensors.warp.isaacgym_warp_mesh_bridge import IsaacGymWarpMeshBridge
import datetime

@torch.jit.script
def copilot_quat_from_rotation_matrix(m):
    """
    将旋转矩阵转换为四元数 (x, y, z, w)
    输入 m: (N, 3, 3)
    """
    batch_size = m.shape[0]
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(batch_size, 4, device=m.device, dtype=m.dtype)
    # 算法：Shepperd's algorithm (避免除以零)
    # Case 1: tr > 0
    mask_pos = tr > 0
    if mask_pos.any():
        S = torch.sqrt(tr[mask_pos] + 1.0) * 2
        q[mask_pos, 3] = 0.25 * S
        q[mask_pos, 0] = (m21[mask_pos] - m12[mask_pos]) / S
        q[mask_pos, 1] = (m02[mask_pos] - m20[mask_pos]) / S
        q[mask_pos, 2] = (m10[mask_pos] - m01[mask_pos]) / S
    # Case 2: tr <= 0
    mask_neg = ~mask_pos
    if mask_neg.any():
        # Find major diagonal element
        cond1 = (m00[mask_neg] > m11[mask_neg]) & (m00[mask_neg] > m22[mask_neg])
        cond2 = ~cond1 & (m11[mask_neg] > m22[mask_neg])
        cond3 = ~cond1 & ~cond2
        # Case 2.1: m00 is largest
        mask_0 = mask_neg.clone()
        mask_0[mask_neg] = cond1
        if mask_0.any():
            S = torch.sqrt(1.0 + m00[mask_0] - m11[mask_0] - m22[mask_0]) * 2
            q[mask_0, 3] = (m21[mask_0] - m12[mask_0]) / S
            q[mask_0, 0] = 0.25 * S
            q[mask_0, 1] = (m01[mask_0] + m10[mask_0]) / S
            q[mask_0, 2] = (m02[mask_0] + m20[mask_0]) / S
        # Case 2.2: m11 is largest
        mask_1 = mask_neg.clone()
        mask_1[mask_neg] = cond2
        if mask_1.any():
            S = torch.sqrt(1.0 + m11[mask_1] - m00[mask_1] - m22[mask_1]) * 2
            q[mask_1, 3] = (m02[mask_1] - m20[mask_1]) / S
            q[mask_1, 0] = (m01[mask_1] + m10[mask_1]) / S
            q[mask_1, 1] = 0.25 * S
            q[mask_1, 2] = (m12[mask_1] + m21[mask_1]) / S
        # Case 2.3: m22 is largest
        mask_2 = mask_neg.clone()
        mask_2[mask_neg] = cond3
        if mask_2.any():
            S = torch.sqrt(1.0 + m22[mask_2] - m00[mask_2] - m11[mask_2]) * 2
            q[mask_2, 3] = (m10[mask_2] - m01[mask_2]) / S
            q[mask_2, 0] = (m02[mask_2] + m20[mask_2]) / S
            q[mask_2, 1] = (m12[mask_2] + m21[mask_2]) / S
            q[mask_2, 2] = 0.25 * S           
    return q

class G1Robot(LeggedRobot):    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # [核心修复] 在 super().__init__ 之前初始化所有特权参数 Buffer
        # 这样即便 play.py 关闭了随机化，compute_observations 也不会因找不到变量而报错
        self.num_envs = cfg.env.num_envs
        self.device = sim_device
        
        # 1. 摩擦力保底：默认 1.0 (CPU 存储，与基类保持一致)
        self.friction_coeffs = torch.ones(self.num_envs, 1, device='cpu', dtype=torch.float)
        
        # 2. 负载质量保底：在 create_sim/_create_envs 期间就会写入，先放 CPU 防止设备不一致
        self.payloads = torch.zeros(self.num_envs, 1, device='cpu', dtype=torch.float)
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 回合级楼梯暴露统计（供 terrain curriculum 升级判定）
        self.episode_stair_conf_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_stair_conf_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_nav_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 回合级运动形态统计（用于诊断平地转圈/横摆）
        self.episode_abs_vx_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_abs_vy_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_abs_wz_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 指令模式掩码：True 表示“随机探索样本”（该样本 yaw 不做视觉覆写）
        self.explore_cmd_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 交替步态记忆：用于抑制“同一只脚长期领步”的局部最优
        self.alt_prev_lead_sign = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.alt_same_lead_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.alt_prev_ds_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # 膝关节索引缓存：用于楼梯段“防直腿前扑”约束
        knee_idx = []
        for name in ("left_knee_joint", "right_knee_joint"):
            if name in self.dof_names:
                knee_idx.append(self.dof_names.index(name))
        if len(knee_idx) == 2:
            self.knee_dof_indices = torch.tensor(knee_idx, dtype=torch.long, device=self.device)
        else:
            self.knee_dof_indices = None

    def _get_explore_task_gate(self, default_weight=0.30):
        """
        探索样本任务降权门控：
        - 普通任务样本: 1.0
        - 探索样本: explore_task_weight (默认 0.30)
        """
        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        if not hasattr(self, "explore_cmd_mask"):
            return gate
        explore_task_weight = float(getattr(self.cfg.rewards, "explore_task_weight", default_weight))
        explore_task_weight = max(0.0, min(explore_task_weight, 1.0))
        return torch.where(
            self.explore_cmd_mask,
            torch.full_like(gate, explore_task_weight),
            gate
        )

    def init_warp_stereo_camera(self):
        """ 初始化 Warp 双目相机，分配显存并绑定内存地址 """
        if not hasattr(self.cfg, 'sensor') or not self.cfg.sensor.stereo_cam.enable:
            return   
        cam_cfg = self.cfg.sensor.stereo_cam
        self.enable_camera_debug = (self.num_envs == 1)
        # 1. 初始化 Warp
        try:
            if not wp.is_initialized(): wp.init()
            wp.config.device = f"cuda:{self.sim_device_id}"
            wp.config.verify_cuda = False
        except Exception as e: 
            print(f"Warp init error: {e}")
        # 2. 准备 PyTorch 端的数据 Buffer
        self.global_cam_pos_buf = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.global_cam_rot_buf = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # 3. 准备地形 Mesh 数据 (处理 Border 偏移)
        if hasattr(self, 'terrain') and self.terrain is not None:
            vertices_np = self.terrain.vertices.astype(np.float32).copy()
            triangles_np = self.terrain.triangles.astype(np.int32)
            border_size = self.cfg.terrain.border_size
            vertices_np[:, 0] -= border_size
            vertices_np[:, 1] -= border_size
        else:
            # Fallback: 如果是平面地形，创建一个简单的地面 Mesh
            s = 20.0 
            vertices_np = np.array([[-s, -s, 0.0], [ s, -s, 0.0], [ s,  s, 0.0], [-s,  s, 0.0]], dtype=np.float32)
            triangles_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        # 4. 创建 IsaacGymWarpMeshBridge 和 Mesh ID
        self.mesh_bridge = IsaacGymWarpMeshBridge(self.gym, self.sim, vertices=vertices_np, triangles=triangles_np, device=self.device)
        mesh_ids = self.mesh_bridge.get_mesh_ids_list(num_envs=self.num_envs)    
        # 5. 创建 WarpStereoCam 对象
        self.stereo_cam = WarpStereoCam(self.num_envs, cam_cfg, mesh_ids, self.device)
        # 6. 分配图像 Buffer (N, 1, H, W) 并传给 Warp
        self.depth_image_buf = torch.zeros(
            (self.num_envs, 1, cam_cfg.height, cam_cfg.width), 
            device=self.device, 
            dtype=torch.float32
        ).contiguous()
        self.stereo_cam.set_image_tensors(pixels=self.depth_image_buf, segmentation_pixels=None)
        # 7. [核心防崩溃] 显式分配 Warp 内存并进行双向绑定
        num_sensors = getattr(cam_cfg, 'num_sensors', 1)
        self.wp_cam_pos = wp.zeros((self.num_envs, num_sensors), dtype=wp.vec3, device=self.device)
        self.wp_cam_rot = wp.zeros((self.num_envs, num_sensors), dtype=wp.quat, device=self.device)    
        # 绑定内存：确保 Python 端持有引用，防止被 GC 回收导致底层非法访问
        self.stereo_cam.camera_position_array = self.wp_cam_pos
        self.stereo_cam.camera_orientation_array = self.wp_cam_rot
        self.stereo_cam.camera_position = self.wp_cam_pos
        self.stereo_cam.camera_orientation = self.wp_cam_rot
        # 8. 锁定相机相对父级 Link 的姿态
        self.cam_parent_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], cam_cfg.parent_link, gymapi.DOMAIN_ACTOR)
        if self.cam_parent_index == -1: self.cam_parent_index = 0
        self.cam_local_pos = torch.tensor(cam_cfg.local_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        pitch_angle = getattr(cam_cfg, 'local_rpy', [0, 0.83, 0])[1]
        # 计算局部旋转四元数
        look_x = np.cos(pitch_angle)
        look_z = -np.sin(pitch_angle)
        look_vec = torch.tensor([look_x, 0.0, look_z], device=self.device, dtype=torch.float32)
        look_vec = look_vec / torch.norm(look_vec)
        right_vec = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=torch.float32)
        down_vec = torch.cross(look_vec, right_vec, dim=-1)
        down_vec = down_vec / torch.norm(down_vec)   
        rot_mat = torch.stack([right_vec, down_vec, look_vec], dim=1)
        self.cam_local_rot = copilot_quat_from_rotation_matrix(rot_mat.unsqueeze(0)).repeat(self.num_envs, 1)
        # 9. 初始化视觉观测 Buffer
        default_dim = cam_cfg.width * cam_cfg.height
        self.visual_feature_dim = getattr(cam_cfg, 'visual_feature_dim', default_dim)
        self.visual_obs_buf = torch.zeros((self.num_envs, self.visual_feature_dim), device=self.device)    
        # 10. 初始化调试射线 (可选保留，不影响性能)
        fov_h_deg = cam_cfg.horizontal_fov_deg
        max_r = cam_cfg.max_range
        aspect_ratio = cam_cfg.width / float(cam_cfg.height)
        angle_h_rad_half = np.deg2rad(fov_h_deg / 2.0)
        angle_v_rad_half = np.arctan(np.tan(angle_h_rad_half) / aspect_ratio)
        dx = np.tan(angle_h_rad_half) * max_r
        dy = np.tan(angle_v_rad_half) * max_r
        rays_local_np = np.array([[-dx, -dy, max_r], [ dx, -dy, max_r], [ dx,  dy, max_r], [-dx,  dy, max_r]], dtype=np.float32)
        self.debug_rays_local = torch.tensor(rays_local_np, device=self.device)
        self.debug_rays_colors = np.array([[1,1,0], [1,1,0], [1,1,0], [1,1,0]], dtype=np.float32)
        print("Warp Stereo Camera Initialized Successfully.")
    # 1. 在类中添加噪声处理函数
    def _apply_depth_noise(self, depth_image_tensor):
        """
        输入: (num_envs, 1, H, W) 或 (num_envs, H, W) 的深度图 tensor
        输出: 添加噪声后的 tensor
        """
        if not self.cfg.sensor.stereo_cam.enable:
            return depth_image_tensor
        # 获取配置
        noise_std = self.cfg.sensor.stereo_cam.noise_level
        dropout_prob = self.cfg.sensor.stereo_cam.dropout_prob
        if noise_std > 0.0:
            # 生成高斯噪声: mean=0, std=noise_std
            noise = torch.randn_like(depth_image_tensor) * noise_std        
            depth_image_tensor += noise
        if dropout_prob > 0.0:
            # 生成一个 mask，概率大于 dropout_prob 的地方保持原样(1)，否则变0
            mask = torch.rand_like(depth_image_tensor) > dropout_prob
            # 将 mask 转为 float 并相乘 (或者直接用布尔索引赋值)
            depth_image_tensor = depth_image_tensor * mask
        # --- 模拟 3: 截断负值和超量程 ---
        depth_image_tensor = torch.clamp(depth_image_tensor, min=0.0, max=self.cfg.sensor.stereo_cam.max_range)
        return depth_image_tensor

    def _update_warp_camera(self):
        """ 更新 Warp 相机：计算姿态，拷贝内存，渲染深度图，后处理噪声 """
        if not hasattr(self, 'stereo_cam') or not self.cfg.sensor.stereo_cam.enable:
            return   
        interval = getattr(self.cfg.sensor.stereo_cam, "update_interval", 1)
        if self.common_step_counter % interval != 0:
            return
        # 1. 使用 PyTorch 计算世界坐标下的相机姿态
        # (利用 GPU 批处理计算，无需 CPU 循环)
        parent_state = self.rigid_body_states_view[:, self.cam_parent_index, :]
        parent_pos = parent_state[:, 0:3]
        parent_rot = parent_state[:, 3:7]     
        from isaacgym.torch_utils import quat_apply, quat_mul
        with torch.no_grad():
            new_global_pos = parent_pos + quat_apply(parent_rot, self.cam_local_pos)
            new_global_rot = quat_mul(parent_rot, self.cam_local_rot)
            self.global_cam_pos_buf.copy_(new_global_pos)
            self.global_cam_rot_buf.copy_(new_global_rot)
        # 2. 将 PyTorch 数据拷贝到 Warp 显存
        # 使用 flatten + view 的方式高效拷贝，避免数据错位
        pos_flat_torch = self.global_cam_pos_buf.reshape(-1).contiguous()
        rot_flat_torch = self.global_cam_rot_buf.reshape(-1).contiguous()   
        # 创建临时 Warp View 进行拷贝
        wp.copy(self.wp_cam_pos.flatten(), wp.from_torch(pos_flat_torch, dtype=wp.float32))
        wp.copy(self.wp_cam_rot.flatten(), wp.from_torch(rot_flat_torch, dtype=wp.float32)) 
        # 确保数据拷贝完成再进行渲染
        wp.synchronize()
        # 3. 执行光线追踪渲染 (C++ Kernel)
        self.stereo_cam.capture()
        # 4. 后处理：加噪声、Clamp、调整维度并存入 visual_obs_buf
        with torch.no_grad():
            cam_cfg = self.cfg.sensor.stereo_cam
            noise_std = float(getattr(cam_cfg, "noise_level", 0.0))
            dropout_prob = float(getattr(cam_cfg, "dropout_prob", 0.0))
            if noise_std > 0.0 or dropout_prob > 0.0:
                noisy_depth = self._apply_depth_noise(self.depth_image_buf.clone())
            else:
                noisy_depth = torch.clamp(self.depth_image_buf, min=0.0, max=cam_cfg.max_range)

            # 训练模式不保存整张深度图到 extras，避免不必要的字典大张量开销
            if getattr(self.cfg.env, "test", False) or self.enable_camera_debug:
                self.extras["depth_camera"] = noisy_depth.squeeze(1)
            # 展平图像特征
            feat = noisy_depth.view(self.num_envs, -1)
            buffer_dim = self.visual_obs_buf.shape[1]       
            # 维度对齐 (防止 feature_dim 设置与图像实际大小不符)
            if feat.shape[1] != buffer_dim:
                if feat.shape[1] > buffer_dim: 
                    feat = feat[:, :buffer_dim]
                else: 
                    feat = torch.cat([feat, torch.zeros((self.num_envs, buffer_dim - feat.shape[1]), device=self.device)], dim=1)         
            self.visual_obs_buf.copy_(feat)
        # 5. 可视化调试 (仅在单环境或 debug 模式下开启)
        if self.enable_camera_debug and hasattr(self, '_debug_visualize_training'):
            self._debug_draw_camera_rays(0, self.global_cam_pos_buf[0], self.global_cam_rot_buf[0])
            self._debug_visualize_training(noisy_depth.squeeze(1))

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
        # current_proprio(50): lin_vel(3), ang_vel(3), gravity(3), commands(3),
        # dof_pos(12), dof_vel(12), prev_actions(12), sin/cos(2)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        noise_vec[12+3*self.num_actions:12+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)   
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        # 兼容初始化顺序：_init_foot() 会再次设置 feet_num，但在此之前
        # 本函数已需要 feet_num 来构造 planner/alternation 缓存。
        if not hasattr(self, "feet_num"):
            self.feet_num = len(self.feet_indices)
        self.obs_history_buf = torch.zeros(self.num_envs, 150, device=self.device, dtype=torch.float)
        # reset 后用当前帧 warm-start 历史观测，避免“全零冷启动”引入抬腿/抖动瞬态
        self.reset_obs_warmstart_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.privileged_physics_params = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        prepare_x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, dtype=torch.float)
        prepare_y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, dtype=torch.float)
        grid_x, grid_y = torch.meshgrid(prepare_x, prepare_y, indexing='ij')
    
        self.measured_points = torch.zeros(grid_x.numel(), 3, device=self.device)
        self.measured_points[:, 0] = grid_x.flatten()
        self.measured_points[:, 1] = grid_y.flatten()
        self.measured_grid_shape = (grid_x.shape[0], grid_x.shape[1])

        self.measured_heights = torch.zeros(self.num_envs, grid_x.numel(), device=self.device)
        # 台阶判定状态机（滞回防抖）：避免边缘地形导致一帧楼梯一帧平地
        self.on_stairs_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 连续楼梯置信度（0~1）：用于软门控，减少奖励/控制硬切
        self.on_stairs_conf_state = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 每步缓存：避免同一仿真步内重复重算楼梯置信度
        self._stairs_raw_conf_cache = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._stairs_raw_conf_cache_step = -1
        self._stairs_conf_cache = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._stairs_conf_cache_step = -1
        # 每步缓存：测高网格 ROI 统计，避免多个奖励函数重复切片/求均值
        self._height_roi_cache_step = -1
        self._roi_center_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_front_near_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_front_far_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_front_commit_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_front_delta = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_underfoot_range = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._roi_local_range = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 每步缓存：楼梯轴向估计（梯度图）
        self._stair_axis_use_x_state = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._stair_axis_cache = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._stair_axis_reliable_cache = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._stair_corner_conf_cache = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._stair_grad_dir_cache = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._stair_axis_cache_step = -1
        # 相位状态机（RUN/SETTLING/HOLD）：
        # - RUN: 有运动意图时按命令驱动步频推进
        # - SETTLING: 收敛到双支撑后再进入 HOLD，避免“摆动中途硬冻结”
        # - HOLD: 零命令静止锁相，切断相位驱动抖腿通道
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.leg_phase = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.phase_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=RUN,1=SETTLING,2=HOLD
        self.phase_ds_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_move_cmd_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.no_cmd_lost_support_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_resume_ramp = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        # 诊断 EMA（用于课程权重闭环，不直接参与动力学）
        self.no_cmd_instability_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        # no_cmd 比例控制 EMA：用于降低重采样时的高频抖动，避免比例控制过冲
        self.no_cmd_rate_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        self.planner_xy_err_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        self.planner_hit8_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        self.planner_hit4_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        # clearance 轨迹跟踪诊断（仅日志，不参与动力学）
        self.clearance_xy_err_ema = torch.zeros(1, dtype=torch.float, device=self.device)
        self.clearance_z_err_ema = torch.zeros(1, dtype=torch.float, device=self.device)

        # Foothold Planner（统一落点目标）缓存：
        # - clearance 使用该目标做安全摆动轨迹跟踪
        # - alternation 使用该目标做触地/换脚结果结算
        self.foothold_plan_active = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.foothold_plan_ttl_steps = torch.zeros((self.num_envs, self.feet_num), dtype=torch.long, device=self.device)
        self.foothold_plan_start_xy = torch.zeros((self.num_envs, self.feet_num, 2), dtype=torch.float, device=self.device)
        self.foothold_plan_target_xy = torch.zeros((self.num_envs, self.feet_num, 2), dtype=torch.float, device=self.device)
        self.foothold_plan_target_z = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_start_h = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_riser_h = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_edge_count = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_dz_goal = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_dz_start = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_conf = torch.zeros((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        self.foothold_plan_top_out = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.foothold_plan_new_event = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.foothold_prev_swing_mask = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.foothold_plan_prev_contact = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.alt_prev_contact_feet = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        # Planner 跟踪奖励事件缓存（与 alternation 解耦）
        self.plan_track_prev_contact = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        # Planner 跟随质量诊断缓存（仅用于日志，不参与奖励）
        self.planner_diag_prev_contact = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        self.planner_diag_touch_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.planner_diag_touch_xy_err_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.planner_diag_touch_z_err_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.planner_diag_touch_xy_err_max = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 交替结算辅助：记录“上一次双支撑是否激活”
        if not hasattr(self, "alt_prev_ds_active"):
            self.alt_prev_ds_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self._init_foot()
        self.init_envs()
        self.session_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.foothold_log = []

    def _get_heights(self, env_ids=None):
        """ 采集机器人下方的地形高度点（上帝视角特权信息） """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), self.measured_points.shape[0], device=self.device)

        # 1. 获取当前机器人的位置和朝向
        res_pos = self.root_states[env_ids, :3]
        res_quat = self.root_states[env_ids, 3:7]
        
        num_envs = len(env_ids)
        num_points = self.measured_points.shape[0]

        # 2. 对齐张量形状以便广播旋转计算 (N*P, 4) 和 (N*P, 3)
        quat_repeated = res_quat.unsqueeze(1).repeat(1, num_points, 1).reshape(-1, 4)
        points_repeated = self.measured_points.unsqueeze(0).repeat(num_envs, 1, 1).reshape(-1, 3)

        # 3. 执行旋转并转换回世界坐标
        points_world = quat_apply(quat_repeated, points_repeated)
        points_world = points_world.view(num_envs, num_points, 3) + res_pos.unsqueeze(1)

        # 4. 转换为地图网格索引
        px = (points_world[:, :, 0] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale
        py = (points_world[:, :, 1] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale
        
        px = px.long().clamp(0, self.terrain.tot_rows - 1)
        py = py.long().clamp(0, self.terrain.tot_cols - 1)

        # 5. 从高度图查表 (使用 _create_trimesh 中定义的 height_samples)
        heights = self.height_samples[px, py].view(num_envs, -1)
        
        return heights * self.cfg.terrain.vertical_scale

    def _estimate_stairs_raw_confidence(self, env_ids=None):
        """融合“边缘查表”和 measured_heights ROI 的原始楼梯置信度（未做时间平滑）。"""
        cur_step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else -1
        if env_ids is None and hasattr(self, "_stairs_raw_conf_cache"):
            if self._stairs_raw_conf_cache_step == cur_step:
                return self._stairs_raw_conf_cache

        n = self.num_envs if env_ids is None else len(env_ids)
        if n == 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)
        conf = torch.zeros(n, dtype=torch.float, device=self.device)
        has_signal = False

        # A) 基于预计算边缘查表的前方边缘密度/跳变置信度
        if getattr(self, "has_stair_edge_lookup", False) and hasattr(self, "height_samples"):
            base_states = self.root_states if env_ids is None else self.root_states[env_ids]
            base_xy = base_states[:, :2]
            yaw = self.rpy[:, 2] if env_ids is None else self.rpy[env_ids, 2]
            fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
            lat = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=1)

            step = float(self.cfg.terrain.horizontal_scale)
            border = float(self.cfg.terrain.border_size)
            vertical_scale = float(self.cfg.terrain.vertical_scale)
            grid_rows, grid_cols = self.height_samples.shape

            if (not hasattr(self, "stair_conf_forward_probes")) or (self.stair_conf_forward_probes.device != self.device):
                self.stair_conf_forward_probes = torch.tensor([0.10, 0.20, 0.30, 0.40], device=self.device, dtype=torch.float32)
                self.stair_conf_lateral_probes = torch.tensor([-0.10, 0.0, 0.10], device=self.device, dtype=torch.float32)
                local = torch.tensor([-1, 0, 1], device=self.device, dtype=torch.long)
                gx, gy = torch.meshgrid(local, local, indexing='ij')
                self.stair_conf_local_dx = gx.reshape(-1)
                self.stair_conf_local_dy = gy.reshape(-1)

            fw = self.stair_conf_forward_probes
            lw = self.stair_conf_lateral_probes
            probe_xy = base_xy.unsqueeze(1).unsqueeze(2) + \
                fwd.unsqueeze(1).unsqueeze(2) * fw.view(1, -1, 1, 1) + \
                lat.unsqueeze(1).unsqueeze(2) * lw.view(1, 1, -1, 1)
            probe_xy = probe_xy.reshape(n, -1, 2)

            use_x_axis = torch.abs(fwd[:, 0]) >= torch.abs(fwd[:, 1])
            axis_sign = torch.where(use_x_axis, torch.sign(fwd[:, 0]), torch.sign(fwd[:, 1]))
            axis_sign = torch.where(torch.abs(axis_sign) < 1e-5, torch.ones_like(axis_sign), axis_sign)

            sample_axis = torch.where(use_x_axis.unsqueeze(1), probe_xy[..., 0], probe_xy[..., 1])
            probe_x_idx = torch.clamp(torch.floor((probe_xy[..., 0] + border) / step), 0, float(grid_rows - 1)).long()
            probe_y_idx = torch.clamp(torch.floor((probe_xy[..., 1] + border) / step), 0, float(grid_cols - 1)).long()

            front_idx = torch.full((n, probe_xy.shape[1]), -1, device=self.device, dtype=torch.int32)
            mask_x_pos = use_x_axis & (axis_sign > 0.0)
            mask_x_neg = use_x_axis & (axis_sign < 0.0)
            mask_y_pos = (~use_x_axis) & (axis_sign > 0.0)
            mask_y_neg = (~use_x_axis) & (axis_sign < 0.0)

            if mask_x_pos.any():
                front_idx[mask_x_pos] = self.stair_edge_front_x_pos[probe_x_idx[mask_x_pos], probe_y_idx[mask_x_pos]]
            if mask_x_neg.any():
                front_idx[mask_x_neg] = self.stair_edge_front_x_neg[probe_x_idx[mask_x_neg], probe_y_idx[mask_x_neg]]
            if mask_y_pos.any():
                front_idx[mask_y_pos] = self.stair_edge_front_y_pos[probe_x_idx[mask_y_pos], probe_y_idx[mask_y_pos]]
            if mask_y_neg.any():
                front_idx[mask_y_neg] = self.stair_edge_front_y_neg[probe_x_idx[mask_y_neg], probe_y_idx[mask_y_neg]]

            front_edge_axis = front_idx.to(torch.float32) * step - border
            front_dist = axis_sign.unsqueeze(1) * (front_edge_axis - sample_axis)
            valid_front = (front_idx >= 0) & (front_dist >= 0.0) & (front_dist <= 0.45)
            near_front = (front_idx >= 0) & (front_dist >= 0.0) & (front_dist <= 0.22)
            edge_density = valid_front.float().mean(dim=1)
            near_density = near_front.float().mean(dim=1)

            base_x_idx = torch.clamp(torch.floor((base_xy[:, 0] + border) / step), 0, float(grid_rows - 1)).long()
            base_y_idx = torch.clamp(torch.floor((base_xy[:, 1] + border) / step), 0, float(grid_cols - 1)).long()
            px = torch.clamp(base_x_idx.unsqueeze(1) + self.stair_conf_local_dx.view(1, -1), 0, grid_rows - 1)
            py = torch.clamp(base_y_idx.unsqueeze(1) + self.stair_conf_local_dy.view(1, -1), 0, grid_cols - 1)
            local_h = self.height_samples[px, py].to(torch.float32) * vertical_scale
            local_range = local_h.amax(dim=1) - local_h.amin(dim=1)

            front_probe_xy = base_xy + fwd * 0.20
            front_x_idx = torch.clamp(torch.floor((front_probe_xy[:, 0] + border) / step), 0, float(grid_rows - 1)).long()
            front_y_idx = torch.clamp(torch.floor((front_probe_xy[:, 1] + border) / step), 0, float(grid_cols - 1)).long()
            h_base = self.height_samples[base_x_idx, base_y_idx].to(torch.float32) * vertical_scale
            h_front = self.height_samples[front_x_idx, front_y_idx].to(torch.float32) * vertical_scale
            front_delta = torch.abs(h_front - h_base)

            density_conf = torch.sigmoid((edge_density - 0.16) / 0.06)
            near_conf = torch.sigmoid((near_density - 0.10) / 0.05)
            range_conf = torch.sigmoid((local_range - 0.035) / 0.008)
            jump_conf = torch.sigmoid((front_delta - 0.012) / 0.004)
            lookup_conf = torch.maximum(torch.maximum(density_conf, near_conf), torch.maximum(range_conf, jump_conf))
            conf = torch.maximum(conf, lookup_conf)
            has_signal = True

        # B) measured_heights ROI 置信度（与旧版兼容，作为补充信号）
        if hasattr(self, "measured_heights"):
            if env_ids is None:
                _, _, _, _, front_delta, _, local_range = self._get_height_roi_features()
            else:
                h = self.measured_heights[env_ids]
                nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
                g = h.view(-1, nx, ny)
                cx, cy = nx // 2, ny // 2
                local = g[:, max(0, cx - 3):min(nx, cx + 4), max(0, cy - 2):min(ny, cy + 3)]
                center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
                front = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]
                local_range = local.amax(dim=(1, 2)) - local.amin(dim=(1, 2))
                front_delta = torch.abs(front.mean(dim=(1, 2)) - center.mean(dim=(1, 2)))
            range_conf = torch.sigmoid((local_range - 0.04) / 0.006)
            front_conf = torch.sigmoid((front_delta - 0.015) / 0.004)
            roi_conf = torch.maximum(range_conf, front_conf)
            conf = torch.maximum(conf, roi_conf)
            has_signal = True

        if not has_signal:
            conf = torch.zeros(n, dtype=torch.float, device=self.device)

        if env_ids is None and hasattr(self, "_stairs_raw_conf_cache"):
            self._stairs_raw_conf_cache.copy_(conf)
            self._stairs_raw_conf_cache_step = cur_step
        return conf

    def _estimate_on_stairs_mask(self, env_ids=None, update_state=True):
        """边缘查表+ROI 融合置信度 + 双阈值滞回的楼梯判定。"""
        raw_conf = self._estimate_stairs_raw_confidence(env_ids)
        enter = raw_conf > 0.42
        exit_ = raw_conf < 0.18

        if hasattr(self, "on_stairs_state"):
            prev = self.on_stairs_state if env_ids is None else self.on_stairs_state[env_ids]
        else:
            prev = torch.zeros_like(enter, dtype=torch.bool)
        mask = torch.where(prev, ~exit_, enter)

        if update_state:
            if env_ids is None:
                self.on_stairs_state = mask
            else:
                self.on_stairs_state[env_ids] = mask

        return mask

    def _estimate_on_stairs_confidence(self, env_ids=None, update_state=True):
        """连续楼梯置信度（0~1），用于软门控，避免状态硬切。"""
        cur_step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else -1
        if env_ids is None and (not update_state) and hasattr(self, "_stairs_conf_cache"):
            if self._stairs_conf_cache_step == cur_step:
                return self._stairs_conf_cache

        stair_conf = self._estimate_stairs_raw_confidence(env_ids)

        beta = 0.2
        if hasattr(self, "on_stairs_conf_state"):
            prev = self.on_stairs_conf_state if env_ids is None else self.on_stairs_conf_state[env_ids]
            conf = (1.0 - beta) * prev + beta * stair_conf
        else:
            conf = stair_conf

        if update_state and hasattr(self, "on_stairs_conf_state"):
            if env_ids is None:
                self.on_stairs_conf_state = conf
            else:
                self.on_stairs_conf_state[env_ids] = conf

        if env_ids is None and hasattr(self, "_stairs_conf_cache"):
            self._stairs_conf_cache.copy_(conf)
            self._stairs_conf_cache_step = cur_step
        elif env_ids is not None and update_state:
            # 子集更新发生时，避免同一步误用全量旧缓存
            if hasattr(self, "_stairs_conf_cache_step"):
                self._stairs_conf_cache_step = -1
            if hasattr(self, "_stairs_raw_conf_cache_step"):
                self._stairs_raw_conf_cache_step = -1

        return conf

    def _get_height_roi_features(self):
        """
        每步缓存测高网格的常用 ROI 统计，供多奖励复用。
        返回：
        center_mean, front_near_mean, front_far_mean, front_commit_mean,
        front_delta, underfoot_range, local_range
        """
        if not hasattr(self, "measured_heights"):
            z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            return z, z, z, z, z, z, z

        cur_step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else -1
        if hasattr(self, "_height_roi_cache_step") and self._height_roi_cache_step == cur_step:
            return (
                self._roi_center_mean,
                self._roi_front_near_mean,
                self._roi_front_far_mean,
                self._roi_front_commit_mean,
                self._roi_front_delta,
                self._roi_underfoot_range,
                self._roi_local_range,
            )

        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        g = self.measured_heights.view(self.num_envs, nx, ny)
        cx, cy = nx // 2, ny // 2

        center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
        front_near = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]
        front_far = g[:, max(0, cx + 5):min(nx, cx + 9), max(0, cy - 1):min(ny, cy + 2)]
        front_commit = g[:, max(0, cx + 2):min(nx, cx + 7), max(0, cy - 1):min(ny, cy + 2)]
        underfoot = g[:, max(0, cx - 2):min(nx, cx + 3), max(0, cy - 2):min(ny, cy + 3)]
        local = g[:, max(0, cx - 3):min(nx, cx + 4), max(0, cy - 2):min(ny, cy + 3)]

        self._roi_center_mean.copy_(center.mean(dim=(1, 2)))
        self._roi_front_near_mean.copy_(front_near.mean(dim=(1, 2)))
        self._roi_front_far_mean.copy_(front_far.mean(dim=(1, 2)))
        self._roi_front_commit_mean.copy_(front_commit.mean(dim=(1, 2)))
        self._roi_front_delta.copy_(torch.abs(self._roi_front_near_mean - self._roi_center_mean))
        self._roi_underfoot_range.copy_(underfoot.amax(dim=(1, 2)) - underfoot.amin(dim=(1, 2)))
        self._roi_local_range.copy_(local.amax(dim=(1, 2)) - local.amin(dim=(1, 2)))
        self._height_roi_cache_step = cur_step

        return (
            self._roi_center_mean,
            self._roi_front_near_mean,
            self._roi_front_far_mean,
            self._roi_front_commit_mean,
            self._roi_front_delta,
            self._roi_underfoot_range,
            self._roi_local_range,
        )

    def _estimate_stair_axis_state(self):
        """
        估计楼梯轴向（带轻量滞回）与角点置信度。
        返回:
        - stair_axis: (N,2), 稳定后的轴向单位向量（x 或 y）
        - axis_reliable: (N,), 单轴可辨识置信度
        - corner_conf: (N,), 直角边/角点置信度
        - grad_dir: (N,2), 梯度方向单位向量
        """
        cur_step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else -1
        if hasattr(self, "_stair_axis_cache_step") and self._stair_axis_cache_step == cur_step:
            return (
                self._stair_axis_cache,
                self._stair_axis_reliable_cache,
                self._stair_corner_conf_cache,
                self._stair_grad_dir_cache,
            )

        stair_axis = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        stair_axis[:, 0] = 1.0
        axis_reliable = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        corner_conf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        grad_dir = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        grad_dir[:, 0] = 1.0

        if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
            base_x = self.root_states[:, 0]
            base_y = self.root_states[:, 1]
            gx = self._lookup_grid_map(self.height_grad_x_map, base_x, base_y, interpolate=True)
            gy = self._lookup_grid_map(self.height_grad_y_map, base_x, base_y, interpolate=True)
            g_abs_x = torch.abs(gx)
            g_abs_y = torch.abs(gy)
            g_min = torch.minimum(g_abs_x, g_abs_y)
            grad_mag = torch.maximum(g_abs_x, g_abs_y)
            anis = torch.abs(g_abs_x - g_abs_y)

            grad_conf = torch.sigmoid((grad_mag - 0.006) / 0.004)
            axis_reliable = grad_conf * torch.sigmoid((anis - 0.004) / 0.003)
            corner_conf = torch.sigmoid((g_min - 0.008) / 0.003) * torch.sigmoid((0.006 - anis) / 0.002)

            if (not hasattr(self, "_stair_axis_use_x_state")) or (self._stair_axis_use_x_state.shape[0] != self.num_envs):
                self._stair_axis_use_x_state = g_abs_x >= g_abs_y
            prev_use_x = self._stair_axis_use_x_state

            dom_margin = 0.002 + 0.004 * corner_conf
            prefer_x = g_abs_x >= (g_abs_y + dom_margin)
            prefer_y = g_abs_y >= (g_abs_x + dom_margin)

            next_use_x = prev_use_x.clone()
            next_use_x = torch.where(prefer_x, torch.ones_like(next_use_x, dtype=torch.bool), next_use_x)
            next_use_x = torch.where(prefer_y, torch.zeros_like(next_use_x, dtype=torch.bool), next_use_x)
            weak_grad_mask = grad_conf < 0.20
            next_use_x = torch.where(weak_grad_mask, prev_use_x, next_use_x)
            self._stair_axis_use_x_state = next_use_x

            stair_axis[:, 0] = next_use_x.float()
            stair_axis[:, 1] = (~next_use_x).float()

            grad_vec = torch.stack([gx, gy], dim=1)
            grad_norm = torch.clamp(torch.norm(grad_vec, dim=1, keepdim=True), min=1e-6)
            grad_dir = grad_vec / grad_norm

        if hasattr(self, "_stair_axis_cache"):
            self._stair_axis_cache.copy_(stair_axis)
            self._stair_axis_reliable_cache.copy_(axis_reliable)
            self._stair_corner_conf_cache.copy_(corner_conf)
            self._stair_grad_dir_cache.copy_(grad_dir)
            self._stair_axis_cache_step = cur_step

        return stair_axis, axis_reliable, corner_conf, grad_dir

    def _get_stair_axis_soft_gain(self, stair_conf, min_gain=0.65):
        """
        楼梯轴向软门控增益：
        - 对齐好: 增益接近 1
        - 对齐差: 正向奖励下调到 [min_gain, 1] 区间
        - 在角点歧义处整体降速/降收益，避免“沿直角边取巧前进”
        """
        if not (hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map")):
            ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            return ones, ones, zeros, zeros

        stair_axis, axis_reliable, corner_conf, _ = self._estimate_stair_axis_state()
        yaw = self.rpy[:, 2]
        body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        axis_align = torch.abs(torch.sum(body_fwd * stair_axis, dim=1))

        align_soft = min_gain + (1.0 - min_gain) * axis_align
        reliable_soft = torch.clamp(0.25 + 0.75 * axis_reliable, min=0.25, max=1.0)
        conf_gate = torch.clamp((stair_conf - 0.08) / 0.30, min=0.0, max=1.0)
        axis_gate = conf_gate * reliable_soft
        # 角点歧义时，降低推进正收益，逼策略先转正轴再拿到高推进分
        corner_amb = corner_conf * (1.0 - axis_reliable)
        corner_gain = 1.0 - 0.40 * corner_amb
        gain = (1.0 - axis_gate * (1.0 - align_soft)) * corner_gain
        return gain, axis_align, axis_reliable, corner_conf

    def _get_progress_velocity_state(self):
        """
        推进速度状态（用于奖励门控）：
        - body_forward: 机体系前向速度（历史兼容）
        - v_axis: 楼梯轴向投影速度（世界系）
        - v_lat: 楼梯轴垂向速度（世界系）
        - axis_mix: 楼梯高置信时逐步从 body_forward 过渡到 v_axis
        """
        body_forward = self.base_lin_vel[:, 0]
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not (hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map")):
            return body_forward, body_forward, zeros, zeros, zeros, zeros, zeros

        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        stair_axis, axis_reliable, corner_conf, _ = self._estimate_stair_axis_state()
        world_vel_xy = self.root_states[:, 7:9]
        v_axis = torch.sum(world_vel_xy * stair_axis, dim=1)
        stair_axis_perp = torch.stack([-stair_axis[:, 1], stair_axis[:, 0]], dim=1)
        v_lat = torch.sum(world_vel_xy * stair_axis_perp, dim=1)

        axis_mix = torch.clamp(
            stair_conf * (0.30 + 0.70 * axis_reliable) * (1.0 - 0.35 * corner_conf),
            min=0.0,
            max=1.0
        )
        progress_vel = (1.0 - axis_mix) * body_forward + axis_mix * v_axis
        return progress_vel, v_axis, v_lat, axis_mix, stair_conf, axis_reliable, corner_conf

    def _get_no_cmd_mask(self, planar_thr=None, yaw_thr=None):
        """统一零指令判据：平面速度命令与偏航命令都接近 0。"""
        if planar_thr is None:
            planar_thr = float(getattr(self.cfg.commands, "no_cmd_planar_thr", 0.05))
        if yaw_thr is None:
            yaw_thr = float(getattr(self.cfg.commands, "no_cmd_yaw_thr", 0.05))
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        no_cmd = (cmd_planar < planar_thr) & (torch.abs(self.commands[:, 2]) < yaw_thr)
        return no_cmd

    def _compute_pose_conditioned_depth_strips(self, depth_img: torch.Tensor):
        """
        轻量姿态条件视觉关注（不改网络结构）：
        - near_strip：更关注近场（脚前台阶边缘）；
        - far_strip：更关注中远场（提前对齐/预判）；
        - 权重由机体姿态与运动状态自适应调整，而非固定行切片。
        """
        # depth_img: [N, H, W]
        h = depth_img.shape[1]
        if (not hasattr(self, "_depth_row_grid")) or (self._depth_row_grid.shape[1] != h) or (self._depth_row_grid.device != self.device):
            self._depth_row_grid = torch.linspace(0.0, 1.0, h, device=self.device, dtype=torch.float32).view(1, h, 1)
        row = self._depth_row_grid

        pitch = self.rpy[:, 1].view(-1, 1, 1)
        roll = self.rpy[:, 0].view(-1, 1, 1)
        tilt = torch.clamp((torch.maximum(torch.abs(pitch), 0.7 * torch.abs(roll)) - 0.06) / 0.16, min=0.0, max=1.0)

        cmd_planar = torch.norm(self.commands[:, :2], dim=1).view(-1, 1, 1)
        move_conf = torch.sigmoid((cmd_planar - 0.10) / 0.05)

        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False).view(-1, 1, 1)
        else:
            stair_conf = torch.zeros((self.num_envs, 1, 1), device=self.device, dtype=torch.float32)

        # 近场关注：楼梯置信高或姿态扰动大时提高；稳定推进时适当给远场更多权重
        near_focus = torch.clamp(0.40 + 0.32 * stair_conf + 0.25 * tilt - 0.20 * move_conf, min=0.18, max=0.95)

        near_center = 0.80 - 0.08 * tilt + 0.05 * (1.0 - move_conf)
        far_center = 0.48 + 0.06 * tilt
        near_sigma = 0.11 + 0.03 * tilt
        far_sigma = 0.14 + 0.02 * (1.0 - tilt)

        near_w = torch.exp(-torch.square(row - near_center) / (2.0 * torch.square(torch.clamp(near_sigma, min=1e-3))))
        far_w = torch.exp(-torch.square(row - far_center) / (2.0 * torch.square(torch.clamp(far_sigma, min=1e-3))))

        # 每个环境内按行归一化
        near_w = near_w / torch.clamp(torch.sum(near_w, dim=1, keepdim=True), min=1e-6)
        far_w = far_w / torch.clamp(torch.sum(far_w, dim=1, keepdim=True), min=1e-6)

        # 增广：让 far strip 也携带一部分 near 信息（避免只看远处忽略脚前边缘）
        far_w = (1.0 - 0.30 * near_focus) * far_w + (0.30 * near_focus) * near_w
        far_w = far_w / torch.clamp(torch.sum(far_w, dim=1, keepdim=True), min=1e-6)

        near_strip = torch.sum(depth_img * near_w, dim=1)
        far_strip = torch.sum(depth_img * far_w, dim=1)
        return near_strip, far_strip

    def _get_cmd_forward_soft_gate(
        self,
        x_thr: float = 0.06,
        planar_thr: float = 0.05,
        x_sigma: float = 0.04,
        planar_sigma: float = 0.04,
        zero_no_cmd: bool = True,
    ):
        """
        统一前进任务软门控：
        - 用 sigmoid 代替二值 hard gate，保留连续梯度；
        - 可选在零指令样本上硬置零，避免静止集污染前进任务奖励。
        """
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        x_sigma = max(float(x_sigma), 1e-4)
        planar_sigma = max(float(planar_sigma), 1e-4)
        cmd_x_soft = torch.sigmoid((self.commands[:, 0] - float(x_thr)) / x_sigma)
        cmd_planar_soft = torch.sigmoid((cmd_planar - float(planar_thr)) / planar_sigma)
        gate = cmd_x_soft * cmd_planar_soft
        if zero_no_cmd:
            gate = torch.where(self._get_no_cmd_mask(), torch.zeros_like(gate), gate)
        return torch.clamp(gate, min=0.0, max=1.0)

    def _get_step_height_target(self, env_ids=None):
        """
        估计每个环境当前课程下的目标台阶高度（米）。
        优先使用 cfg.terrain.stair_height_levels，其次退回默认映射。
        """
        if env_ids is None:
            levels = self.terrain_levels
        else:
            levels = self.terrain_levels[env_ids]

        rows = float(max(self.cfg.terrain.num_rows, 1))
        default_step_h = 0.05 + 0.1 * (levels.float() / rows)

        height_levels = getattr(self.cfg.terrain, "stair_height_levels", None)
        if not (isinstance(height_levels, (list, tuple)) and len(height_levels) > 0):
            return default_step_h

        if (not hasattr(self, "_stair_height_levels_tensor")) or \
           (self._stair_height_levels_tensor.device != self.device):
            self._stair_height_levels_tensor = torch.tensor(
                [float(v) for v in height_levels],
                dtype=torch.float,
                device=self.device
            )

        n = int(self._stair_height_levels_tensor.numel())
        if n <= 1:
            return torch.full_like(default_step_h, float(self._stair_height_levels_tensor[0].item()))

        # terrain_levels 仍按 cfg.terrain.num_rows 分级；映射到 stair_height_levels 维度
        row_idx = torch.clamp(levels.long(), min=0, max=max(int(self.cfg.terrain.num_rows) - 1, 0))
        map_idx = torch.round(
            row_idx.float() * float(n - 1) / max(float(self.cfg.terrain.num_rows - 1), 1.0)
        ).long()
        map_idx = torch.clamp(map_idx, min=0, max=n - 1)
        return self._stair_height_levels_tensor[map_idx]

    def _get_feet_corner_heights(self, env_ids=None):
        """
        获取每只脚四角点对应的地形高度，返回形状: (N, feet_num, 4)。
        """
        if env_ids is None:
            feet_xy = self.feet_pos[:, :, :2]
            feet_quat = self.rigid_body_states_view[:, self.feet_indices, 3:7]
            yaw = self.rpy[:, 2]
            num = self.num_envs
        else:
            feet_xy = self.feet_pos[env_ids, :, :2]
            feet_quat = self.rigid_body_states_view[env_ids][:, self.feet_indices, 3:7]
            yaw = self.rpy[env_ids, 2]
            num = len(env_ids)

        front_len = 0.12
        back_len = 0.05
        half_width = 0.03

        x_axis = torch.zeros(num * self.feet_num, 3, device=self.device, dtype=torch.float32)
        x_axis[:, 0] = 1.0
        foot_fwd_3d = quat_apply(feet_quat.reshape(-1, 4), x_axis).view(num, self.feet_num, 3)
        foot_fwd = foot_fwd_3d[:, :, :2]
        fwd_norm = torch.norm(foot_fwd, dim=-1, keepdim=True)
        base_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1).unsqueeze(1)
        foot_fwd = torch.where(fwd_norm > 1e-4, foot_fwd / (fwd_norm + 1e-6), base_fwd)
        foot_lat = torch.stack([-foot_fwd[:, :, 1], foot_fwd[:, :, 0]], dim=-1)

        off_fb = torch.tensor([front_len, front_len, -back_len, -back_len], device=self.device, dtype=torch.float32).view(1, 1, 4, 1)
        off_lr = torch.tensor([half_width, -half_width, half_width, -half_width], device=self.device, dtype=torch.float32).view(1, 1, 4, 1)
        corners_xy = feet_xy.unsqueeze(2) + foot_fwd.unsqueeze(2) * off_fb + foot_lat.unsqueeze(2) * off_lr
        corners_h = self._get_terrain_heights(
            None,
            corners_xy[..., 0].reshape(-1),
            corners_xy[..., 1].reshape(-1),
            interpolate=False
        ).view(num, self.feet_num, 4)
        return corners_h

    def _get_feet_corner_top_heights(self, env_ids=None):
        """
        以四角点地形高度的最大值估计每只脚所在台阶高度（用户需求：用四角最高 z 判定）。
        返回形状: (N, feet_num)
        """
        corners_h = self._get_feet_corner_heights(env_ids=env_ids)
        return corners_h.max(dim=-1).values

    def _get_touchdown_contact_quality(self, contact_now):
        """
        触地质量过滤（用于规划失活和触地事件监督）：
        - 要有足够垂向支撑力；
        - 侧向冲击比例不能过高（避免踢台阶侧边被当成有效落地）。
        """
        forces = self.contact_forces[:, self.feet_indices, :]
        force_xy = torch.norm(forces[:, :, :2], dim=2)
        force_z = torch.clamp(forces[:, :, 2], min=0.0)
        side_ratio = force_xy / (force_z + 1e-6)
        min_fz = float(getattr(self.cfg.rewards, "touchdown_min_fz", 1.2))
        side_ratio_max = float(getattr(self.cfg.rewards, "touchdown_side_ratio_max", 0.85))
        quality = contact_now & (force_z > min_fz) & (side_ratio < side_ratio_max)
        return quality, side_ratio

    def _update_foothold_planner(self):
        """
        统一落点规划器（事件触发，DS->SS 时锁存）：
        - 无边缘: 使用速度预测落点
        - 有边缘: 以 rect 最优深度 9cm 为锚点修正落点
        - 需跨多级但边缘不足: 回退到最后可用边缘（top-out 过渡）
        """
        if not hasattr(self, "leg_phase"):
            return

        if hasattr(self, "foothold_plan_new_event"):
            self.foothold_plan_new_event[:] = False
        if not hasattr(self, "foothold_plan_ttl_steps"):
            self.foothold_plan_ttl_steps = torch.zeros(
                (self.num_envs, self.feet_num), dtype=torch.long, device=self.device
            )
        if not hasattr(self, "foothold_plan_start_xy"):
            self.foothold_plan_start_xy = torch.zeros(
                (self.num_envs, self.feet_num, 2), dtype=torch.float, device=self.device
            )
        self.foothold_plan_ttl_steps = torch.where(
            self.foothold_plan_ttl_steps > 0,
            self.foothold_plan_ttl_steps - 1,
            self.foothold_plan_ttl_steps,
        )

        swing_start = 0.55
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
        touchdown_quality, _ = self._get_touchdown_contact_quality(contact_now)
        if not hasattr(self, "foothold_plan_prev_contact"):
            self.foothold_plan_prev_contact = contact_now.clone()
        liftoff_event = (~contact_now) & self.foothold_plan_prev_contact

        cmd_planar_all = torch.norm(self.commands[:, :2], dim=1)
        forward_task = (self.commands[:, 0] > 0.06) & (cmd_planar_all > 0.05)
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            _, _, _, _, front_delta, _, local_range = self._get_height_roi_features()
            near_front_conf = torch.sigmoid((front_delta - 0.008) / 0.004)
            near_local_conf = torch.sigmoid((local_range - 0.030) / 0.008)
            stair_task = torch.maximum(stair_conf, torch.maximum(near_front_conf, near_local_conf)) > 0.16
        else:
            stair_task = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        planner_task_env = forward_task & stair_task

        # 非任务环境下，禁用规划激活，避免无效目标参与奖励。
        self.foothold_plan_active = torch.where(
            planner_task_env.unsqueeze(1),
            self.foothold_plan_active,
            torch.zeros_like(self.foothold_plan_active),
        )
        self.foothold_plan_ttl_steps = torch.where(
            planner_task_env.unsqueeze(1),
            self.foothold_plan_ttl_steps,
            torch.zeros_like(self.foothold_plan_ttl_steps),
        )
        non_task_env = ~planner_task_env
        if non_task_env.any():
            self.foothold_plan_conf[non_task_env] = 0.0
            self.foothold_plan_edge_count[non_task_env] = 0.0
            self.foothold_plan_top_out[non_task_env] = False
            self.foothold_plan_new_event[non_task_env] = False
            self.foothold_plan_dz_goal[non_task_env] = 0.0
            self.foothold_plan_dz_start[non_task_env] = 0.0
            self.foothold_plan_target_xy[non_task_env] = self.feet_pos[non_task_env, :, :2]
            self.foothold_plan_target_z[non_task_env] = self.feet_pos[non_task_env, :, 2]
        # 脚触地后关闭该脚激活标记：
        # 采用“摆动->支撑切换”事件（上一拍在摆动、本拍触地）立即失活，
        # 下一次离地事件会覆盖并重新写入新目标。
        touchdown_raw = contact_now & self.foothold_prev_swing_mask & (self.leg_phase < 0.55)
        touchdown_valid = touchdown_raw & touchdown_quality
        # 兜底：若先发生侧碰等低质量接触，进入支撑早期后仍需结算旧目标，防止 active 悬挂太久。
        fallback_phase_max = float(getattr(self.cfg.rewards, "touchdown_fallback_phase_max", 0.18))
        touchdown_fallback = self.foothold_plan_active & contact_now & (self.leg_phase < fallback_phase_max)
        touchdown_stable = touchdown_valid | touchdown_fallback
        self.foothold_plan_active = torch.where(
            touchdown_stable,
            torch.zeros_like(self.foothold_plan_active),
            self.foothold_plan_active,
        )
        self.foothold_plan_ttl_steps = torch.where(
            touchdown_stable,
            torch.zeros_like(self.foothold_plan_ttl_steps),
            self.foothold_plan_ttl_steps,
        )

        # 离地即预测：只要“支撑->摆动”切换就更新本次目标，不再依赖 hold/latch。
        enter_swing = liftoff_event

        swing_now = ~contact_now
        self.foothold_prev_swing_mask = swing_now.clone()

        if not enter_swing.any():
            self.foothold_plan_active = self.foothold_plan_active & (self.foothold_plan_ttl_steps > 0)
            self.foothold_plan_prev_contact = contact_now.clone()
            return

        period = 0.8
        swing_duration = (1.0 - swing_start) * period
        swing_ttl_steps = max(int(round(float(swing_duration) / max(float(self.dt), 1e-6))), 1)
        edge_h_thr = float(getattr(self.cfg.terrain, "edge_height_threshold", 0.03))
        target_depth = 0.09
        depth_min, depth_max = 0.05, 0.13
        sample_num = 9
        alphas = torch.linspace(0.0, 1.0, sample_num, device=self.device, dtype=torch.float)

        for swing_foot in range(self.feet_num):
            support_foot = 1 - swing_foot
            env_ids = torch.where(enter_swing[:, swing_foot])[0]
            if env_ids.numel() == 0:
                continue

            start_xy = self.feet_pos[env_ids, swing_foot, :2]
            world_vel = self.root_states[env_ids, 7:9]
            yaw = self.rpy[env_ids, 2]
            body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)

            vel_norm = torch.norm(world_vel, dim=1, keepdim=True)
            vel_dir_raw = torch.where(vel_norm > 0.05, world_vel / (vel_norm + 1e-6), body_fwd)
            # 反后滑保护：前进任务下若真实速度方向后滑，方向回拉到机体前向
            vel_forward_proj = torch.sum(vel_dir_raw * body_fwd, dim=1, keepdim=True)
            cmd_forward_conf = torch.sigmoid((self.commands[env_ids, 0].unsqueeze(1) - 0.06) / 0.04)
            backward_conf = cmd_forward_conf * torch.sigmoid((-vel_forward_proj - 0.02) / 0.06)
            vel_dir_mix = (1.0 - backward_conf) * vel_dir_raw + backward_conf * body_fwd
            vel_dir_norm = torch.norm(vel_dir_mix, dim=1, keepdim=True)
            vel_dir = torch.where(vel_dir_norm > 1e-5, vel_dir_mix / (vel_dir_norm + 1e-6), body_fwd)
            step = float(self.cfg.terrain.horizontal_scale)
            border = float(self.cfg.terrain.border_size)
            # 规划方向改为“台阶边缘法线方向”，避免速度方向漂移导致落点斜切。
            if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
                stair_axis_all, _, _, grad_dir_all = self._estimate_stair_axis_state()
                stair_axis = stair_axis_all[env_ids]
                grad_dir = grad_dir_all[env_ids]
            else:
                stair_axis = torch.zeros_like(body_fwd)
                stair_axis[:, 0] = 1.0
                grad_dir = stair_axis.clone()
            use_x_axis = stair_axis[:, 0] >= stair_axis[:, 1]
            grad_axis_proj = torch.sum(grad_dir * stair_axis, dim=1)
            cmd_axis_proj = torch.sum(self.commands[env_ids, :2] * stair_axis, dim=1)
            body_axis_proj = torch.sum(body_fwd * stair_axis, dim=1)
            axis_sign_hint = torch.where(torch.abs(cmd_axis_proj) > 0.03, cmd_axis_proj, body_axis_proj)
            axis_sign_raw = torch.where(torch.abs(grad_axis_proj) > 0.02, grad_axis_proj, axis_sign_hint)
            axis_sign = torch.sign(axis_sign_raw)
            axis_sign = torch.where(torch.abs(axis_sign) < 1e-5, torch.ones_like(axis_sign), axis_sign)
            dir_unit = stair_axis * axis_sign.unsqueeze(1)

            # 以四角最高 z 估计台阶级差：
            # - 同级(含死区)：level_gap = 0 -> 跨 1 级
            # - 相差 k 级：level_gap = k -> 跨 k+1 级
            corners_h = self._get_feet_corner_heights(env_ids=env_ids)
            top_h = corners_h.max(dim=-1).values
            support_contact_now = contact_now[env_ids, support_foot]
            support_anchor_xy_raw = self.feet_pos[env_ids, support_foot, :2]
            if hasattr(self, "foothold_plan_target_xy") and hasattr(self, "foothold_plan_conf"):
                support_plan_xy = self.foothold_plan_target_xy[env_ids, support_foot, :2]
                support_plan_valid = self.foothold_plan_conf[env_ids, support_foot] > 0.05
            else:
                support_plan_xy = support_anchor_xy_raw
                support_plan_valid = torch.zeros_like(support_contact_now)
            # 新规则：
            # 1) 对侧脚处于支撑 -> 用对侧当前脚位作为台阶基座；
            # 2) 对侧脚不在支撑且有有效预测 -> 用对侧预测落点作为基座；
            # 3) 否则回退到对侧当前脚位。
            use_plan_anchor = (~support_contact_now) & support_plan_valid
            support_anchor_xy = torch.where(
                use_plan_anchor.unsqueeze(1),
                support_plan_xy,
                support_anchor_xy_raw
            )
            support_h_now = top_h[:, support_foot]
            # 摆动脚高度改为四角最高值：
            # 预测只在离地切换瞬间触发，直接用最高点更符合“当前所处台阶级”。
            swing_h = top_h[:, swing_foot]
            # 台阶等级基准与基座来源一致：
            # - 对侧支撑时：用对侧当前四角最高点；
            # - 对侧摆动时：用对侧预测落点处地形高度。
            support_h_plan = self._get_terrain_heights(
                None,
                support_anchor_xy[:, 0],
                support_anchor_xy[:, 1],
                interpolate=False
            ).view(-1)
            support_h = torch.where(support_contact_now, support_h_now, support_h_plan)
            step_h = torch.clamp(self._get_step_height_target(env_ids=env_ids), min=0.05, max=0.20)
            level_diff_signed = support_h - swing_h
            same_level_thr = torch.clamp(0.30 * step_h, min=0.022, max=0.055)
            level_diff_eff = torch.where(
                torch.abs(level_diff_signed) <= same_level_thr,
                torch.zeros_like(level_diff_signed),
                level_diff_signed,
            )
            level_gap = torch.round(
                torch.clamp(level_diff_eff / (step_h + 1e-4), min=0.0, max=3.0)
            ).long()
            # 统一规则：无论基座来自哪种情况，都只尝试“基座 +1 级”。
            # 若找不到上一级边缘，再回退到基座同级（后续 lookup 分支处理）。
            desired_cross = torch.ones_like(level_gap)

            cmd_planar = torch.norm(self.commands[env_ids, :2], dim=1)
            move_intent = torch.sigmoid((self.commands[env_ids, 0] - 0.08) / 0.05) * \
                          torch.sigmoid((cmd_planar - 0.08) / 0.05)
            # 回调前探硬度，降低低速扰动时的过冲。
            v_min = 0.30
            virtual_speed = vel_norm.squeeze(1) + move_intent * torch.clamp(v_min - vel_norm.squeeze(1), min=0.0)
            # 楼梯前探距离下限：多级跨边保持；单级跨边按命令强度自适应，减小弱前进时过冲。
            stair_tread = float(getattr(self.cfg.terrain, "stair_step_width", 0.30))
            probe_floor_default = torch.clamp(
                desired_cross.float() * stair_tread + 0.06,
                min=0.32,
                max=0.78
            )
            strong_forward = (self.commands[env_ids, 0] > 0.22) & (cmd_planar > 0.12)
            single_cross_floor = torch.where(
                strong_forward,
                torch.full_like(probe_floor_default, stair_tread + target_depth),
                torch.full_like(probe_floor_default, stair_tread + 0.03),
            )
            probe_floor = torch.where(desired_cross <= 1, single_cross_floor, probe_floor_default)
            probe_floor = torch.clamp(probe_floor, min=0.30, max=0.78)
            probe_dist = torch.maximum(virtual_speed * swing_duration, probe_floor)
            raw_target_xy = start_xy + dir_unit * probe_dist.unsqueeze(1)

            # 与 rect 统一：优先使用“预计算台阶边缘查表”来做跨边计数与边缘定位
            if getattr(self, "has_stair_edge_lookup", False):
                grid_rows, grid_cols = self.height_samples.shape
                if (not hasattr(self, "rect_lateral_offsets")) or (self.rect_lateral_offsets.device != self.device):
                    self.rect_lateral_offsets = torch.tensor([-1, 0, 1], device=self.device, dtype=torch.long)
                lat_offsets = self.rect_lateral_offsets
                k = int(lat_offsets.numel())
                def _lookup_front_edge_axis(sample_xy):
                    sample_axis = torch.where(use_x_axis, sample_xy[:, 0], sample_xy[:, 1])
                    x_idx = torch.clamp(torch.floor((sample_xy[:, 0] + border) / step), 0, float(grid_rows - 1)).long()
                    y_idx = torch.clamp(torch.floor((sample_xy[:, 1] + border) / step), 0, float(grid_cols - 1)).long()
                    x_idx_x = x_idx.unsqueeze(-1).expand(-1, k)
                    y_idx_x = torch.clamp(y_idx.unsqueeze(-1) + lat_offsets.view(1, -1), 0, grid_cols - 1)
                    x_idx_y = torch.clamp(x_idx.unsqueeze(-1) + lat_offsets.view(1, -1), 0, grid_rows - 1)
                    y_idx_y = y_idx.unsqueeze(-1).expand(-1, k)

                    front_idx_cand = torch.full((len(env_ids), k), -1, device=self.device, dtype=torch.int32)
                    mask_x_pos = use_x_axis & (axis_sign > 0.0)
                    mask_x_neg = use_x_axis & (axis_sign < 0.0)
                    mask_y_pos = (~use_x_axis) & (axis_sign > 0.0)
                    mask_y_neg = (~use_x_axis) & (axis_sign < 0.0)
                    if mask_x_pos.any():
                        front_idx_cand[mask_x_pos] = self.stair_edge_front_x_pos[x_idx_x[mask_x_pos], y_idx_x[mask_x_pos]]
                    if mask_x_neg.any():
                        front_idx_cand[mask_x_neg] = self.stair_edge_front_x_neg[x_idx_x[mask_x_neg], y_idx_x[mask_x_neg]]
                    if mask_y_pos.any():
                        front_idx_cand[mask_y_pos] = self.stair_edge_front_y_pos[x_idx_y[mask_y_pos], y_idx_y[mask_y_pos]]
                    if mask_y_neg.any():
                        front_idx_cand[mask_y_neg] = self.stair_edge_front_y_neg[x_idx_y[mask_y_neg], y_idx_y[mask_y_neg]]

                    edge_axis_cand = front_idx_cand.to(torch.float32) * step - border
                    edge_dist_cand = axis_sign.unsqueeze(-1) * (edge_axis_cand - sample_axis.unsqueeze(-1))
                    valid = (front_idx_cand >= 0) & (edge_dist_cand >= 0.0)
                    inf = torch.full_like(edge_dist_cand, 1e6)
                    best_dist = torch.min(torch.where(valid, edge_dist_cand, inf), dim=-1).values
                    has_edge_local = best_dist < 1e5
                    best_axis = sample_axis + axis_sign * best_dist
                    return has_edge_local, best_axis

                # 规则化目标：
                # - 台阶级别/边缘来源：对侧脚；
                # - 目标点落在当前摆动脚的“本脚横向通道”上，避免左右脚目标互相串道；
                # - 找不到上一级边缘（如最高台阶）=> 回退到“对侧脚当前等级”并投影到摆动脚通道，再 +9cm。
                has_support_up_edge, support_up_edge_axis = _lookup_front_edge_axis(support_anchor_xy)
                support_same_level_axis = torch.where(use_x_axis, support_anchor_xy[:, 0], support_anchor_xy[:, 1])
                # 关键修复：边缘轴向值来自支撑脚，但横向坐标保持摆动脚通道，防止可视化“交叉落点”。
                swing_lane_xy = start_xy.clone()
                swing_lane_edge_xy = swing_lane_xy.clone()
                swing_lane_edge_xy[:, 0] = torch.where(use_x_axis, support_up_edge_axis, swing_lane_edge_xy[:, 0])
                swing_lane_edge_xy[:, 1] = torch.where(~use_x_axis, support_up_edge_axis, swing_lane_edge_xy[:, 1])
                swing_lane_same_level_xy = swing_lane_xy.clone()
                swing_lane_same_level_xy[:, 0] = torch.where(use_x_axis, support_same_level_axis, swing_lane_same_level_xy[:, 0])
                swing_lane_same_level_xy[:, 1] = torch.where(~use_x_axis, support_same_level_axis, swing_lane_same_level_xy[:, 1])
                edge_xy = torch.where(has_support_up_edge.unsqueeze(1), swing_lane_edge_xy, swing_lane_same_level_xy)
                has_exact = has_support_up_edge
                edge_count = torch.where(
                    has_support_up_edge,
                    torch.ones_like(desired_cross),
                    torch.zeros_like(desired_cross)
                )
            else:
                # 兜底：无查表时退回旧版“连线高度差判边缘”
                line_pts = start_xy.unsqueeze(1) + (raw_target_xy - start_xy).unsqueeze(1) * alphas.view(1, sample_num, 1)
                line_h = self._get_terrain_heights(
                    None,
                    line_pts[..., 0].reshape(-1),
                    line_pts[..., 1].reshape(-1),
                    interpolate=False
                ).view(len(env_ids), sample_num)
                dh = line_h[:, 1:] - line_h[:, :-1]
                edge_mask = dh > edge_h_thr
                edge_count = edge_mask.sum(dim=1)
                edge_cumsum = torch.cumsum(edge_mask.int(), dim=1)
                desired_exact = edge_mask & (edge_cumsum == desired_cross.unsqueeze(1))
                has_exact = desired_exact.any(dim=1)
                last_k = torch.clamp(edge_count, min=1)
                desired_last = edge_mask & (edge_cumsum == last_k.unsqueeze(1))
                chosen_mask = torch.where(has_exact.unsqueeze(1), desired_exact, desired_last)
                chosen_seg_idx = torch.argmax(chosen_mask.int(), dim=1)
                edge_alpha = alphas[torch.clamp(chosen_seg_idx + 1, max=sample_num - 1)]
                edge_xy = start_xy + (raw_target_xy - start_xy) * edge_alpha.unsqueeze(1)

            # rect 锚点修正：edge + 9cm（并夹到 rect 可行域）
            depth = torch.clamp(torch.full((len(env_ids),), target_depth, device=self.device), min=depth_min, max=depth_max)
            anchor_xy = edge_xy + dir_unit * depth.unsqueeze(1)

            # 目标恒为“对应边缘 + 9cm”。
            target_xy = anchor_xy

            target_z = self._get_terrain_heights(
                None,
                target_xy[:, 0],
                target_xy[:, 1],
                interpolate=False
            ).view(-1)

            # 对最终目标连线采样高度，记录路径峰值；边缘数量优先沿用查表计数。
            line_pts_final = start_xy.unsqueeze(1) + (target_xy - start_xy).unsqueeze(1) * alphas.view(1, sample_num, 1)
            line_h_final = self._get_terrain_heights(
                None,
                line_pts_final[..., 0].reshape(-1),
                line_pts_final[..., 1].reshape(-1),
                interpolate=False
            ).view(len(env_ids), sample_num)
            line_dh_final = line_h_final[:, 1:] - line_h_final[:, :-1]
            if getattr(self, "has_stair_edge_lookup", False):
                edge_count_final = edge_count.to(torch.float32)
            else:
                edge_count_final = (line_dh_final > edge_h_thr).sum(dim=1).to(torch.float32)
            path_peak_h = line_h_final.amax(dim=1)
            start_h = line_h_final[:, 0]
            end_h = line_h_final[:, -1]
            riser_h = torch.clamp(path_peak_h - torch.maximum(start_h, end_h), min=0.0, max=0.20)

            dz_goal = target_z - support_h
            support_z_ref = self.feet_pos[env_ids, support_foot, 2]
            dz_start = self.feet_pos[env_ids, swing_foot, 2] - support_z_ref
            no_edge_current = (~has_exact)
            top_out = no_edge_current & (desired_cross > 0)
            conf_raw = torch.where(has_exact, torch.full_like(target_z, 0.95), torch.full_like(target_z, 0.72))
            task_env_local = planner_task_env[env_ids]
            edge_count_final = torch.where(task_env_local, edge_count_final, torch.zeros_like(edge_count_final))
            conf = torch.where(task_env_local, conf_raw, torch.zeros_like(conf_raw))
            top_out = top_out & task_env_local

            # 这版按事件“直接写入当次规划结果”，避免旧目标残留干扰当前步态。
            self.foothold_plan_target_xy[env_ids, swing_foot] = target_xy
            self.foothold_plan_target_z[env_ids, swing_foot] = target_z
            self.foothold_plan_start_xy[env_ids, swing_foot] = start_xy
            self.foothold_plan_start_h[env_ids, swing_foot] = start_h
            self.foothold_plan_riser_h[env_ids, swing_foot] = riser_h
            self.foothold_plan_edge_count[env_ids, swing_foot] = edge_count_final
            self.foothold_plan_dz_goal[env_ids, swing_foot] = dz_goal
            self.foothold_plan_dz_start[env_ids, swing_foot] = dz_start
            self.foothold_plan_conf[env_ids, swing_foot] = conf
            self.foothold_plan_top_out[env_ids, swing_foot] = top_out
            self.foothold_plan_new_event[env_ids, swing_foot] = task_env_local
            self.foothold_plan_active[env_ids, swing_foot] = task_env_local
            ttl_vals = torch.where(
                task_env_local,
                torch.full((len(env_ids),), swing_ttl_steps, dtype=torch.long, device=self.device),
                torch.zeros((len(env_ids),), dtype=torch.long, device=self.device),
            )
            self.foothold_plan_ttl_steps[env_ids, swing_foot] = ttl_vals

        self.foothold_plan_active = self.foothold_plan_active & (self.foothold_plan_ttl_steps > 0)

        self.foothold_plan_prev_contact = contact_now.clone()

    def _process_rigid_body_props(self, props, env_id):
        """ 拦截并记录每个环境随机生成的负载增量 """
        # 先保存随机化之前的质量（基准）
        base_mass = props[0].mass
        
        # 1. 运行基类逻辑进行随机化
        props = super()._process_rigid_body_props(props, env_id)
        
        # 2. 记录负载增量
        if self.cfg.domain_rand.randomize_base_mass:
            # 记录这次随机多出来的公斤数
            self.payloads[env_id] = props[0].mass - base_mass
            
        return props

    def init_envs(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return
        if not hasattr(self, 'terrain') or self.terrain is None:
            return
            
        # =========================================================
        # 👑 [终极地形出生点修复]
        # 彻底覆盖底层残缺的 _get_env_origins，让它乖乖出生在坑底！
        # =========================================================
        
        # 1. 强制把每个环境的原点，绑定到真实的地形坑底中心！
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        
        # 2. 计算最终的世界坐标 XY
        init_pos = self.cfg.init_state.pos
        offset_x, offset_y, spawn_offset_z = init_pos[0], init_pos[1], init_pos[2]
        
        total_x = self.env_origins[:, 0] + offset_x
        total_y = self.env_origins[:, 1] + offset_y
        
        # 3. 转换为网格坐标 (Grid Indices)
        horizontal_scale = self.cfg.terrain.horizontal_scale
        vertical_scale = self.cfg.terrain.vertical_scale
        border_size = self.cfg.terrain.border_size      
        
        grid_x = ((total_x + border_size) / horizontal_scale).long().clamp(0, self.terrain.tot_rows - 1)
        grid_y = ((total_y + border_size) / horizontal_scale).long().clamp(0, self.terrain.tot_cols - 1)   
        
        # 4. 查表获取脚下的真实地形高度
        raw_heights = self.terrain.height_field_raw[grid_x.cpu().numpy(), grid_y.cpu().numpy()]
        terrain_heights = torch.tensor(raw_heights, device=self.device) * vertical_scale
        self.env_origins[:, 2] = terrain_heights    
        
        # 5. [最关键的修改]：不仅要修正 Z 轴，必须强制覆盖 X 和 Y 轴！
        self.root_states[:, 0] = total_x
        self.root_states[:, 1] = total_y
        self.root_states[:, 2] = terrain_heights + spawn_offset_z    

        # 课程统计基线同步：
        # init_envs() 会在初始化期重定位 root_states，若不同时更新 episode_start_xy，
        # 首回合 distance 统计会出现“虚高位移”，从而误触发 terrain move_up。
        if hasattr(self, "episode_start_xy"):
            self.episode_start_xy[:] = self.root_states[:, :2]
        if hasattr(self, "episode_max_distance"):
            self.episode_max_distance[:] = 0.0
        
        # 6. 刷新物理引擎
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))    
        print(f"[G1 Info] 出生点已完美修复: 强制绑定到地形中心坑底!")
    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def compute_reward(self):
        """ 
        [最终全能版奖励课程]
        涵盖：姿态、步态精修、运动质量、导航引导
        """
        # # 1. 获取进度与 dt
        # if self.cfg.env.test:
        #     # 如果是 play.py 测试模式，强制拉满地形进度！
        #     # 激活所有高级别的高抬腿惩罚、脚掌平整惩罚和视觉引导！
        #     terrain_progress = 1.0
        # else:
        #     # 训练模式下，正常随机器人平均所在等级增加难度
        # avg_level = torch.mean(self.terrain_levels.float())
        # max_level = self.cfg.terrain.num_rows - 1
        # terrain_progress = torch.clip(avg_level / max_level, 0.0, 1.0)
        # 与 terrain.py 口径一致：difficulty = terrain_level / num_rows
        rows = float(self.cfg.terrain.num_rows)
        terrain_progress_raw = torch.clamp(self.terrain_levels.float() / rows, 0.0, 1.0)
        # 课程进度仍归一化到 [0, 1]，并以“可达最大难度”作为 1.0
        terrain_progress_max = max((rows - 1.0) / rows, 1e-6)
        terrain_progress = torch.clamp(terrain_progress_raw / terrain_progress_max, 0.0, 1.0)

        # 双阶段课程：后期项延后启动并采用缓升，避免“技能未稳就全面变严”
        late_stage_start_level = float(getattr(self.cfg.terrain, "late_stage_start_level", 5.0))
        late_stage_start_raw = late_stage_start_level / rows
        late_stage_den = max(terrain_progress_max - late_stage_start_raw, 1e-6)
        late_stage_progress_raw = torch.clamp(
            (terrain_progress_raw - late_stage_start_raw) / late_stage_den,
            min=0.0,
            max=1.0
        )
        late_stage_pow = float(getattr(self.cfg.terrain, "late_stage_progress_pow", 1.6))
        late_stage_progress = torch.pow(late_stage_progress_raw, max(late_stage_pow, 1.0))
        dt = self.dt
        # 混合课程：通用稳定项用全局进度，楼梯专项项保留 per-env 进度
        global_progress = torch.mean(terrain_progress)
        global_late_stage_progress = torch.mean(late_stage_progress)
        if not hasattr(self, "late_stage_progress_ema"):
            self.late_stage_progress_ema = torch.zeros(1, device=self.device)
        ema_alpha = 0.015
        self.late_stage_progress_ema = (1.0 - ema_alpha) * self.late_stage_progress_ema + \
            ema_alpha * global_late_stage_progress.detach().unsqueeze(0)
        late_stage_progress_smooth = torch.clamp(
            0.65 * late_stage_progress + 0.35 * self.late_stage_progress_ema.squeeze(0),
            min=0.0,
            max=1.0
        )

        def safe_update(name, val, per_env=False):
            if name not in self.reward_scales:
                return
            if per_env:
                self.reward_scales[name] = val * dt
            else:
                if torch.is_tensor(val):
                    val = val.item() if val.numel() == 1 else torch.mean(val).item()
                self.reward_scales[name] = float(val) * dt

        # === 闭环自适应信号（来自 callback 诊断 EMA） ===
        if hasattr(self, "no_cmd_instability_ema"):
            no_cmd_inst = torch.clamp(self.no_cmd_instability_ema.squeeze(0), min=0.0, max=2.2)
        else:
            no_cmd_inst = torch.tensor(0.0, device=self.device)
        # 失稳闭环保持“可用但不打满”：
        # 过早饱和会让 no_cmd 惩罚长期主导优化，挤占推进主任务梯度。
        no_cmd_boost = torch.clamp((no_cmd_inst - 0.42) / 1.45, min=0.0, max=0.45)

        if hasattr(self, "planner_xy_err_ema"):
            planner_xy_err_ema = torch.clamp(self.planner_xy_err_ema.squeeze(0), min=0.0, max=0.45)
        else:
            planner_xy_err_ema = torch.tensor(0.20, device=self.device)
        if hasattr(self, "planner_hit8_ema"):
            planner_hit8_ema = torch.clamp(self.planner_hit8_ema.squeeze(0), min=0.0, max=1.0)
        else:
            planner_hit8_ema = torch.tensor(0.0, device=self.device)

        planner_miss_boost = torch.clamp((planner_xy_err_ema - 0.14) / 0.30, min=0.0, max=0.75)
        planner_hit_gate = torch.clamp((planner_hit8_ema - 0.12) / 0.40, min=0.0, max=1.0)
        # Planner 与 alternation 真解耦：
        # 两者权重不再互相调制，仅保留诊断项用于观察命中/失配趋势。
        planner_follow_gate = torch.ones_like(planner_hit_gate)
        planner_focus_boost = torch.ones_like(planner_miss_boost)

        safe_update("tracking_lin_vel", 1.72 - 0.12 * late_stage_progress_smooth, per_env=True)
        # 稳定项：stand_still 已改为正向稳定分，课程后期与失稳阶段适度抬权
        safe_update(
            "stand_still",
            0.24 + 0.14 * late_stage_progress_smooth + 0.12 * no_cmd_boost,
            per_env=True
        )
        self.dynamic_base_height_target = 0.72 + 0.03 * float(global_progress.item())
        safe_update("orientation", -0.25 - 0.35 * late_stage_progress_smooth, per_env=True)
        # safe_update("pelvis_height", 0.08 + 0.12 * late_stage_progress, per_env=True)
        safe_update("pelvis_height", 0.10 + 0.12 * late_stage_progress_smooth, per_env=True)
        safe_update("stair_alignment", -0.20 - 0.36 * late_stage_progress_smooth, per_env=True)
        safe_update("ang_vel_xy", -0.022 - 0.030 * late_stage_progress_smooth, per_env=True)
        ######
        safe_update("approach_stairs", 0.44 + 0.18 * (1.0 - late_stage_progress_smooth), per_env=True)
        safe_update("tracking_ang_vel", 1.10 + 0.10 * terrain_progress, per_env=True)
        safe_update("contact", 0.10 - 0.02 * late_stage_progress_smooth, per_env=True)
        # 保持“敢迈第一步”信号到中后期，减少台阶前停顿等待
        safe_update(
            "first_step_commit",
            0.32 + 0.16 * (1.0 - late_stage_progress_smooth) + 0.06 * planner_miss_boost,
            per_env=True
        )
        safe_update("lin_vel_z", -0.82 + 0.10 * late_stage_progress_smooth, per_env=True)
        # 动作变化率渐进约束：前期更松便于学会推进，后期再抑制抖动/拖步
        safe_update("action_rate", -0.012 - 0.020 * late_stage_progress_smooth, per_env=True)
        # 台阶安全专项：保持主导但避免压过速度/姿态主任务
        safe_update("foot_support_rect", 0.06 + 0.24 * torch.pow(late_stage_progress_smooth, 1.35), per_env=True)
        safe_update(
            "stair_clearance",
            0.16 + 0.28 * late_stage_progress_smooth,
            per_env=True
        )
        safe_update("feet_stumble", -0.40 - 0.80 * late_stage_progress_smooth, per_env=True)
        safe_update("hip_pos", -0.22 - 0.16 * late_stage_progress_smooth, per_env=True)
        # 交替节奏项（解耦后独立课程）
        safe_update(
            "stair_alternation",
            0.18 + 0.50 * late_stage_progress_smooth,
            per_env=True
        )
        # Planner 跟踪项（解耦后独立课程，略高于交替节奏，优先保证踩中预测点）
        safe_update(
            "planner_tracking",
            0.52 + 0.63 * late_stage_progress_smooth,
            per_env=True
        )
        ###########
        # safe_update("approach_stairs", 0.138 + 0.078 * (1.0 - late_stage_progress), per_env=True)
        # safe_update("tracking_ang_vel", 0.920 + 0.080 * terrain_progress, per_env=True)
        # safe_update("contact", 0.082 + 0.060 * late_stage_progress, per_env=True)
        # safe_update("first_step_commit", 0.040 + 0.030 * (1.0 - late_stage_progress), per_env=True)
        # safe_update("lin_vel_z", -0.900 + 0.216 * late_stage_progress, per_env=True)
        # safe_update("action_rate", -0.024 - 0.012 * late_stage_progress, per_env=True)
        # safe_update("foot_support_rect", -0.220 - 1.780 * torch.pow(late_stage_progress, 1.50), per_env=True)
        # safe_update("stair_clearance", 0.48 + 0.48 * late_stage_progress, per_env=True)
        # safe_update("feet_stumble", -0.210 - 0.780 * late_stage_progress, per_env=True)
        # safe_update("stair_alternation", 3.300 + 3.300 * late_stage_progress, per_env=True)

        self.current_terrain_progress = terrain_progress
        self.current_late_stage_progress = late_stage_progress_smooth
        
        # 【核心修正】：直接塞进全局 extras，不依赖机器人的 reset，每一步都实时更新！
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["Metrics/metrics_terrain_progress"] = torch.mean(terrain_progress).item()
        self.extras["Diagnostics/sched_no_cmd_boost"] = float(no_cmd_boost.item())
        self.extras["Diagnostics/sched_planner_miss_boost"] = float(planner_miss_boost.item())
        self.extras["Diagnostics/sched_planner_hit_gate"] = float(planner_hit_gate.item())
        self.extras["Diagnostics/sched_planner_follow_gate"] = float(torch.mean(planner_follow_gate).item())
        self.extras["Diagnostics/sched_planner_focus_boost"] = float(torch.mean(planner_focus_boost).item())
        return super().compute_reward()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        manual_cmd_override = bool(getattr(self.cfg.env, "manual_cmd_override", False))
        self.update_feet_state()
        self._update_full_contact_rate_metric()
        # 默认给角速度跟踪奖励一个“可用目标”，避免在无视觉帧时目标缺失
        if not hasattr(self, "visual_target_yaw_vel"):
            self.visual_target_yaw_vel = torch.zeros(self.num_envs, device=self.device)
        self.visual_target_yaw_vel[:] = self.commands[:, 2]
        # 唯一真值：执行与奖励共用的 yaw 跟踪目标
        if not hasattr(self, "yaw_track_target"):
            self.yaw_track_target = torch.zeros(self.num_envs, device=self.device)
        self.yaw_track_target[:] = self.commands[:, 2]

        # 1. 测高雷达更新
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            
        # 2. 视觉老司机：每一步物理仿真后，重新渲染图像并进行方向盘纠偏
        if hasattr(self, 'stereo_cam') and self.cfg.sensor.stereo_cam.enable:
            self._update_warp_camera()

        # 手动模式下仅保留视觉观测更新，不覆写外部下发的速度指令
        if hasattr(self, 'stereo_cam') and self.cfg.sensor.stereo_cam.enable and not (self.cfg.env.test and manual_cmd_override):
            # Play/Test 模式下禁用 explore 掩码，避免 resampling_time 很大时
            # 初始一次重采样把 env 标成探索样本后长期误关视觉纠偏。
            if hasattr(self, "explore_cmd_mask") and (not self.cfg.env.test):
                explore_mask = self.explore_cmd_mask
            else:
                explore_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            non_explore_conf = (~explore_mask).float()

            # 前进控制权软门控：仅在“以前进为主”时放大视觉纠偏
            # 这样侧移/后退阶段主要服从手动/策略本身，不被视觉硬拽。
            forward_cmd = self.commands[:, 0]
            cmd_planar = torch.norm(self.commands[:, :2], dim=1)
            forward_x_conf = torch.sigmoid((forward_cmd - 0.10) / 0.05)  # 0~1
            lateral_keep_conf = torch.sigmoid((0.15 - torch.abs(self.commands[:, 1])) / 0.05)
            # 不再使用 yaw_keep_conf，避免 resample 的随机 yaw 直接影响视觉纠偏权重。
            # 视觉接管由“前进任务意图 + 视觉场景置信度”共同决定，而不是由随机 yaw 决定。
            # 探索样本不做视觉 yaw 覆写：保持其随机 yaw 指令用于探索
            # 零/弱前进意图时不注入视觉 yaw，避免“零指令也被视觉带着走”
            drive_intent_conf = torch.clamp((forward_cmd - 0.06) / 0.10, min=0.0, max=1.0)
            forward_conf = forward_x_conf * lateral_keep_conf * drive_intent_conf * non_explore_conf

            cam_cfg = self.cfg.sensor.stereo_cam
            # depth_img shape: (num_envs, 20, 32) 完美匹配 height=20, width=32
            depth_img = self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width)

            # a. 姿态条件视觉区域：自适应提取近场/远场条带
            near_strip, far_strip = self._compute_pose_conditioned_depth_strips(depth_img)
            # far_strip：用于看前方开阔度（是否可持续推进）
            forward_strip = far_strip
            # 平地/登顶开阔度（软判定，0~1）
            forward_mean_depth = torch.mean(forward_strip, dim=1)
            open_conf = torch.sigmoid((forward_mean_depth - 2.6) / 0.2)

            # b. near_strip：用于脚前落足安全/左右对齐
            depth_strip = near_strip

            # c. 【视区精准三等分】：针对 width = 32 进行切分 (11 + 10 + 11 = 32)
            left_depth = torch.mean(depth_strip[:, 0:11], dim=1)
            center_depth = torch.mean(depth_strip[:, 11:21], dim=1)
            right_depth = torch.mean(depth_strip[:, 21:32], dim=1)
            # 分母保护：避免深度过小导致比值放大，触发 escape_conf 偶发虚高
            depth_floor = 0.10
            left_depth = torch.clamp(left_depth, min=depth_floor)
            center_depth = torch.clamp(center_depth, min=depth_floor)
            right_depth = torch.clamp(right_depth, min=depth_floor)

            # d. 【正常对齐逻辑：寻找平衡】
            # 右浅左深（朝左歪）-> 右转 (-yaw) 纠偏
            depth_diff = right_depth - left_depth
            # 对齐死区软化：阈值附近不再硬清零，而是渐进减弱纠偏
            align_gain = torch.sigmoid((torch.abs(depth_diff) - 0.10) / 0.03)
            align_err = depth_diff * align_gain

            # 视觉可见台阶置信度：用于“是否下放视觉控制权”的判定
            side_depth = 0.5 * (left_depth + right_depth)
            seen_depth_thr = float(getattr(cam_cfg, "vision_seen_depth_thresh", 1.25))
            seen_depth_sigma = max(float(getattr(cam_cfg, "vision_seen_depth_sigma", 0.18)), 1e-4)
            seen_edge_thr = float(getattr(cam_cfg, "vision_seen_edge_thresh", 0.10))
            seen_edge_sigma = max(float(getattr(cam_cfg, "vision_seen_edge_sigma", 0.03)), 1e-4)
            center_near_conf = torch.sigmoid((seen_depth_thr - center_depth) / seen_depth_sigma)
            center_step_conf = torch.sigmoid((torch.abs(side_depth - center_depth) - seen_edge_thr) / seen_edge_sigma)
            visual_seen_conf = torch.maximum(center_near_conf, center_step_conf)

            # e. 【紧急逃逸硬门控（可选）】
            # 去掉常规 escape 混合，仅在“非常近 + 强不对称”时启用紧急逃逸。
            escape_direction = torch.where(left_depth > right_depth,
                                          torch.full_like(align_err, 1.0),
                                          torch.full_like(align_err, -1.0))
            center_gap = torch.clamp(center_depth - 0.5 * (left_depth + right_depth), min=0.0)
            escape_strength = torch.clamp(torch.abs(depth_diff) * 2.0 + center_gap * 0.6, min=0.0, max=1.2)
            escape_yaw = escape_direction * escape_strength
            enable_emergency_escape = bool(getattr(cam_cfg, "enable_emergency_escape", True))
            emergency_center_depth_thresh = float(getattr(cam_cfg, "emergency_center_depth_thresh", 0.25))
            emergency_depth_diff_thresh = float(getattr(cam_cfg, "emergency_depth_diff_thresh", 0.30))
            if enable_emergency_escape:
                emergency_escape = (center_depth < emergency_center_depth_thresh) & \
                                   (torch.abs(depth_diff) > emergency_depth_diff_thresh)
                raw_yaw = torch.where(emergency_escape, escape_yaw, align_err)
            else:
                raw_yaw = align_err

            # g. 【纯视觉 Sim2Real 修复：封堵侧身走”长廊错觉”】
            # 彻底抛弃上帝视角 (self.terrain_types == 2)！
            # 直接利用机器人的局部测高雷达 (大范围高程图) 判断所处地形。
            stair_conf = self._estimate_on_stairs_confidence(update_state=True)
            # 梯度图轴向辅助：进入楼梯且轴可靠时，把视觉 yaw 轻推向“楼梯主轴方向”，
            # 避免对着金字塔直角边斜切前进。
            if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
                stair_axis, axis_reliable, corner_conf, _ = self._estimate_stair_axis_state()
                axis_heading = torch.atan2(stair_axis[:, 1], stair_axis[:, 0])
                yaw_now = self.rpy[:, 2]
                axis_err = wrap_to_pi(axis_heading - yaw_now)
                axis_err_flip = wrap_to_pi(axis_heading + math.pi - yaw_now)
                axis_err = torch.where(torch.abs(axis_err) <= torch.abs(axis_err_flip), axis_err, axis_err_flip)
                axis_cmd = torch.clamp(1.6 * axis_err, min=-0.8, max=0.8)
                axis_gate = torch.clamp(
                    stair_conf * (0.30 + 0.70 * axis_reliable) * (1.0 - 0.55 * corner_conf),
                    min=0.0,
                    max=1.0
                )
                raw_yaw = (1.0 - axis_gate) * raw_yaw + axis_gate * axis_cmd

            # 👑【动态控制权移交 - 软权重版】
            # 目标：前期强视觉纠偏，后期逐步下放给策略网络，但不完全放弃视觉安全约束。
            max_open_release = float(getattr(cam_cfg, "max_open_release", 0.45))
            forward_keep = torch.clamp(1.0 - 0.6 * forward_conf, min=0.4, max=1.0)
            terrain_or_seen_conf = torch.maximum(stair_conf, visual_seen_conf)
            override_w = max_open_release * torch.pow(open_conf, 1.5) * (1.0 - terrain_or_seen_conf) * forward_keep

            # 课程化控制权下放：terrain_progress 越高，视觉直接接管比例越低
            handover_end_gain = float(getattr(cam_cfg, "visual_handover_end_gain", 0.65))
            if hasattr(self, "current_terrain_progress"):
                progress = self.current_terrain_progress
                if not torch.is_tensor(progress):
                    progress = torch.full((self.num_envs,), float(progress), device=self.device)
                elif progress.ndim == 0:
                    progress = progress.repeat(self.num_envs)
                else:
                    progress = progress.to(self.device)
            else:
                progress = torch.zeros(self.num_envs, device=self.device)
            visual_gain = 1.0 - (1.0 - handover_end_gain) * torch.clamp(progress, min=0.0, max=1.0)
            desired_yaw = visual_gain * (1.0 - override_w) * raw_yaw

            # 限制极速，防止打方向盘过猛摔倒 (建议保持 1.0，保证尖角逃生速度)
            target_yaw_vel = torch.clamp(desired_yaw, -1.0, 1.0)
            # 记录“视觉原始纠偏目标”，供角速度跟踪奖励使用
            old_cmd_yaw = self.commands[:, 2]
            self.visual_target_yaw_vel[:] = torch.where(explore_mask, old_cmd_yaw, target_yaw_vel)
            alpha = 0.2
            base_smoothed = (1.0 - alpha) * old_cmd_yaw + alpha * target_yaw_vel

            # 反向优先：当目标与当前指令方向相反且目标幅值有效时，直接切到目标方向，
            # 避免“需要左转时仍在右转减速”的长拖尾。
            reverse_mask = (old_cmd_yaw * target_yaw_vel < 0.0) & (torch.abs(target_yaw_vel) > 0.008)
            smoothed_yaw_vel = torch.where(reverse_mask, target_yaw_vel, base_smoothed)

            # 死区仅对非反向场景生效；反向场景保留小幅反向指令以完成过零切换。
            smoothed_yaw_vel = torch.where((torch.abs(smoothed_yaw_vel) < 0.03) & (~reverse_mask),
                                           torch.zeros_like(smoothed_yaw_vel),
                                           smoothed_yaw_vel)

            # Play/Test 手动策略：仅“纯前进命令”时启用视觉 yaw 纠偏，
            # 其他命令（侧移/后退/手动转向）优先服从外部输入。
            old_cmd_yaw = self.commands[:, 2]
            if self.cfg.env.test:
                manual_priority_mask = (forward_cmd <= 0.06) | \
                                       (torch.abs(self.commands[:, 1]) > 0.08) | \
                                       (torch.abs(old_cmd_yaw) > 0.08)
            else:
                manual_priority_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            visual_enable_mask = (~manual_priority_mask) & (~explore_mask)
            visual_enable_f = visual_enable_mask.float()

            # h. 【执行指令覆盖】
            if self.cfg.commands.heading_command:
                # Heading 模式：按前进置信度渐进接管，避免指令阈值附近硬切
                visual_heading = self.commands[:, 3] + (forward_conf * visual_enable_f) * smoothed_yaw_vel * self.dt
                self.commands[:, 3] = torch.where(visual_enable_mask, visual_heading, self.commands[:, 3])
                # heading 模式下，奖励跟踪“本步视觉纠偏后的角速度目标”
                self.yaw_track_target[:] = torch.where(visual_enable_mask, smoothed_yaw_vel, self.commands[:, 2])
            else:
                # 角速度模式下，直接接管并覆盖底层随机角速度
                # 零指令时强制回零，彻底切断历史 yaw 残留导致的慢速自旋
                idle_planar_mask = self._get_no_cmd_mask()
                idle_decay = torch.where(
                    idle_planar_mask,
                    torch.full_like(old_cmd_yaw, 0.35),
                    torch.full_like(old_cmd_yaw, 0.85)
                )
                idle_yaw = self.commands[:, 2] * idle_decay
                idle_yaw = torch.where(torch.abs(idle_yaw) < 0.02,
                                       torch.zeros_like(idle_yaw),
                                       idle_yaw)
                blended_yaw = forward_conf * smoothed_yaw_vel + (1.0 - forward_conf) * idle_yaw
                visual_cmd_yaw = torch.clamp(blended_yaw, -1.0, 1.0)
                # 全零命令时强制回零，避免历史视觉指令残留
                visual_cmd_yaw = torch.where(idle_planar_mask, torch.zeros_like(visual_cmd_yaw), visual_cmd_yaw)
                self.commands[:, 2] = torch.where(visual_enable_mask, visual_cmd_yaw, old_cmd_yaw)
                # yaw 模式下，奖励与执行完全对齐
                self.yaw_track_target[:] = torch.where(
                    idle_planar_mask & visual_enable_mask,
                    torch.zeros_like(self.commands[:, 2]),
                    self.commands[:, 2]
                )

        # no_cmd 控制：每步保持零命令，避免残留小命令把站立样本拉回慢速行走。
        if not (self.cfg.env.test and manual_cmd_override):
            no_cmd_mask = self._get_no_cmd_mask()
            if no_cmd_mask.any():
                self.commands[no_cmd_mask, 0] = 0.0
                self.commands[no_cmd_mask, 1] = 0.0
                if self.cfg.commands.heading_command:
                    forward = quat_apply(self.base_quat[no_cmd_mask], self.forward_vec[no_cmd_mask])
                    current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                    self.commands[no_cmd_mask, 3] = current_heading
                else:
                    self.commands[no_cmd_mask, 2] = 0.0
                if hasattr(self, "visual_target_yaw_vel"):
                    self.visual_target_yaw_vel[no_cmd_mask] = 0.0
                if hasattr(self, "yaw_track_target"):
                    self.yaw_track_target[no_cmd_mask] = 0.0

        # 回合级楼梯暴露统计（供课程升级逻辑读取）
        if hasattr(self, "episode_stair_conf_sum"):
            stair_conf_step = self._estimate_on_stairs_confidence(update_state=False)
            nav_mask = (torch.norm(self.commands[:, :2], dim=1) > 0.05)
            self.episode_stair_conf_sum += stair_conf_step * nav_mask.float()
            self.episode_stair_conf_max = torch.maximum(self.episode_stair_conf_max, stair_conf_step)
            self.episode_nav_steps += nav_mask.long()
            self.episode_abs_vx_sum += torch.abs(self.base_lin_vel[:, 0]) * nav_mask.float()
            self.episode_abs_vy_sum += torch.abs(self.base_lin_vel[:, 1]) * nav_mask.float()
            self.episode_abs_wz_sum += torch.abs(self.base_ang_vel[:, 2]) * nav_mask.float()

        # 诊断指标（独立栏目，不放在奖励关键词下）
        if not hasattr(self, "extras"):
            self.extras = {}
        no_cmd_mask_diag = self._get_no_cmd_mask()
        left_contact_diag = self.contact_forces[:, self.feet_indices[0], 2] > 1.2
        right_contact_diag = self.contact_forces[:, self.feet_indices[1], 2] > 1.2
        single_support_diag = left_contact_diag ^ right_contact_diag
        double_flight_diag = (~left_contact_diag) & (~right_contact_diag)
        no_cmd_count = torch.sum(no_cmd_mask_diag.float())
        if no_cmd_count > 0:
            no_cmd_single_support_rate = torch.sum((single_support_diag & no_cmd_mask_diag).float()) / no_cmd_count
            no_cmd_double_flight_rate = torch.sum((double_flight_diag & no_cmd_mask_diag).float()) / no_cmd_count
            no_cmd_yaw_rate_mean = torch.mean(torch.abs(self.base_ang_vel[no_cmd_mask_diag, 2]))
        else:
            no_cmd_single_support_rate = torch.tensor(0.0, device=self.device)
            no_cmd_double_flight_rate = torch.tensor(0.0, device=self.device)
            no_cmd_yaw_rate_mean = torch.tensor(0.0, device=self.device)
        move_cmd_mask_diag = ~no_cmd_mask_diag
        move_cmd_count = torch.sum(move_cmd_mask_diag.float())
        if move_cmd_count > 0:
            move_double_flight_rate = torch.sum((double_flight_diag & move_cmd_mask_diag).float()) / move_cmd_count
        else:
            move_double_flight_rate = torch.tensor(0.0, device=self.device)
        self.extras["Diagnostics/no_cmd_rate"] = torch.mean(no_cmd_mask_diag.float()).item()
        self.extras["Diagnostics/no_cmd_single_support_rate"] = no_cmd_single_support_rate.item()
        self.extras["Diagnostics/no_cmd_double_flight_rate"] = no_cmd_double_flight_rate.item()
        self.extras["Diagnostics/move_double_flight_rate"] = move_double_flight_rate.item()
        self.extras["Diagnostics/no_cmd_yaw_rate_mean"] = no_cmd_yaw_rate_mean.item()
        cmd_planar_diag = torch.norm(self.commands[:, :2], dim=1)
        self.extras["Diagnostics/cmd_planar_zero_rate"] = torch.mean((cmd_planar_diag < 0.05).float()).item()
        if hasattr(self, "no_cmd_instability_ema"):
            yaw_inst = torch.clamp((no_cmd_yaw_rate_mean - 0.20) / 0.80, min=0.0, max=1.5)
            inst_step = torch.clamp(
                1.35 * no_cmd_single_support_rate
                + 1.70 * no_cmd_double_flight_rate
                + 0.60 * yaw_inst,
                min=0.0,
                max=2.2,
            )
            no_cmd_rate_val = float(torch.mean(no_cmd_mask_diag.float()).item())
            if no_cmd_rate_val > 0.01:
                ema_alpha = 0.04
                self.no_cmd_instability_ema = (1.0 - ema_alpha) * self.no_cmd_instability_ema + \
                    ema_alpha * inst_step.detach().unsqueeze(0)
            else:
                self.no_cmd_instability_ema = 0.985 * self.no_cmd_instability_ema
            self.extras["Diagnostics/no_cmd_instability_ema"] = float(self.no_cmd_instability_ema.item())

        # print(f"前进速度：{forward_mask}")
        # print(f"[纠偏] yaw_vel: {self.visual_target_yaw_vel.mean().item():.3f}, base_ang: {self.base_ang_vel[:, 2].mean().item():.3f}, cmd: {self.commands[:, 2].mean().item():.3f}")
        # 3. 更新腿部相位（命令驱动 + 双支撑锁相状态机）
        # 目标：
        # - 有运动意图时按命令驱动步频推进；
        # - 零命令时先收敛到双支撑再锁相，避免“摆动中途冻结”引发抖腿/摔倒。
        run_mode = 0
        settling_mode = 1
        hold_mode = 2

        ds_stable_need = 3
        move_resume_need = 4
        settle_freq = float(getattr(self.cfg.commands, "no_cmd_settling_phase_rate", 0.00))
        run_freq_base = 0.85
        run_freq_min = 0.75
        run_freq_max = 2.20
        # RUN 恢复步频爬升时长：略放慢，降低 reset/停转走时的瞬时失稳
        resume_steps = 18

        no_cmd_phase = self._get_no_cmd_mask()
        move_intent = ~no_cmd_phase
        left_contact_phase = self.contact_forces[:, self.feet_indices[0], 2] > 1.2
        right_contact_phase = self.contact_forces[:, self.feet_indices[1], 2] > 1.2
        double_support_phase = left_contact_phase & right_contact_phase
        double_flight_phase = (~left_contact_phase) & (~right_contact_phase)

        prev_mode = self.phase_mode
        next_mode = prev_mode.clone()

        run_mask = prev_mode == run_mode
        run_to_settle = run_mask & no_cmd_phase
        next_mode = torch.where(run_to_settle, torch.full_like(next_mode, settling_mode), next_mode)
        if run_to_settle.any():
            self.phase_ds_stable_steps[run_to_settle] = 0
            # 进入 no_cmd 收敛态时强制回到双支撑起点，避免从摆动段切入导致两腿同步乱动
            self.phase[run_to_settle] = 0.0
            if hasattr(self, "phase_left"):
                self.phase_left[run_to_settle] = 0.0
            if hasattr(self, "phase_right"):
                self.phase_right[run_to_settle] = 0.0
            if hasattr(self, "leg_phase"):
                self.leg_phase[run_to_settle] = 0.0
        # no_cmd 下双脚腾空采用“连续帧确认”回收，避免每帧硬重置相位造成控制跳变。
        if not hasattr(self, "no_cmd_lost_support_steps"):
            self.no_cmd_lost_support_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        no_cmd_lost_support_now = no_cmd_phase & double_flight_phase
        self.no_cmd_lost_support_steps = torch.where(
            no_cmd_lost_support_now,
            self.no_cmd_lost_support_steps + 1,
            torch.zeros_like(self.no_cmd_lost_support_steps),
        )
        no_cmd_lost_support_active = self.no_cmd_lost_support_steps >= 2
        no_cmd_lost_support_trigger = self.no_cmd_lost_support_steps == 2
        next_mode = torch.where(no_cmd_lost_support_active, torch.full_like(next_mode, settling_mode), next_mode)
        if no_cmd_lost_support_trigger.any():
            self.phase_ds_stable_steps[no_cmd_lost_support_trigger] = 0
            self.phase[no_cmd_lost_support_trigger] = 0.0
            if hasattr(self, "phase_left"):
                self.phase_left[no_cmd_lost_support_trigger] = 0.0
            if hasattr(self, "phase_right"):
                self.phase_right[no_cmd_lost_support_trigger] = 0.0
            if hasattr(self, "leg_phase"):
                self.leg_phase[no_cmd_lost_support_trigger] = 0.0

        settle_mask = prev_mode == settling_mode
        settle_to_run = settle_mask & move_intent
        next_mode = torch.where(settle_to_run, torch.full_like(next_mode, run_mode), next_mode)
        if settle_to_run.any():
            self.phase_resume_ramp[settle_to_run] = 0.0
            self.phase_ds_stable_steps[settle_to_run] = 0

        settle_active = settle_mask & (~settle_to_run)
        if settle_active.any():
            ds_inc_mask = settle_active & no_cmd_phase & double_support_phase
            self.phase_ds_stable_steps[ds_inc_mask] += 1
            self.phase_ds_stable_steps[settle_active & (~ds_inc_mask)] = 0
            settle_to_hold = settle_active & (self.phase_ds_stable_steps >= ds_stable_need)
            next_mode = torch.where(settle_to_hold, torch.full_like(next_mode, hold_mode), next_mode)
            if settle_to_hold.any():
                self.phase_ds_stable_steps[settle_to_hold] = 0
                self.phase[settle_to_hold] = 0.0

        hold_mask = prev_mode == hold_mode
        if hold_mask.any():
            hold_move_mask = hold_mask & move_intent
            self.phase_move_cmd_steps[hold_move_mask] += 1
            self.phase_move_cmd_steps[hold_mask & (~move_intent)] = 0
            hold_to_run = hold_mask & (self.phase_move_cmd_steps >= move_resume_need)
            next_mode = torch.where(hold_to_run, torch.full_like(next_mode, run_mode), next_mode)
            if hold_to_run.any():
                self.phase_resume_ramp[hold_to_run] = 0.0
                self.phase_move_cmd_steps[hold_to_run] = 0

        self.phase_move_cmd_steps[next_mode != hold_mode] = 0
        self.phase_mode = next_mode

        cmd_planar_phase = torch.norm(self.commands[:, :2], dim=1)
        cmd_forward_phase = torch.clamp(self.commands[:, 0], min=0.0)
        freq_target = run_freq_base + 1.10 * cmd_forward_phase + 0.25 * cmd_planar_phase
        freq_target = torch.clamp(freq_target, min=run_freq_min, max=run_freq_max)

        run_mask_now = self.phase_mode == run_mode
        if run_mask_now.any():
            ramp_step = 1.0 / max(float(resume_steps), 1.0)
            self.phase_resume_ramp[run_mask_now] = torch.clamp(
                self.phase_resume_ramp[run_mask_now] + ramp_step, min=0.0, max=1.0
            )
        self.phase_resume_ramp[~run_mask_now] = 0.0

        run_freq = run_freq_base + self.phase_resume_ramp * (freq_target - run_freq_base)
        phase_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        phase_rate = torch.where(self.phase_mode == run_mode, run_freq, phase_rate)
        phase_rate = torch.where(self.phase_mode == settling_mode, torch.full_like(phase_rate, settle_freq), phase_rate)

        self.phase = torch.remainder(self.phase + phase_rate * self.dt, 1.0)
        hold_mask_now = self.phase_mode == hold_mode
        if hold_mask_now.any():
            self.phase[hold_mask_now] = 0.0

        offset = 0.5
        self.phase_left = self.phase
        phase_right_default = torch.remainder(self.phase + offset, 1.0)
        # HOLD/SETTLING 模式都使用双脚同相，帮助 no_cmd 更快收敛到稳定双支撑。
        settling_mask_now = self.phase_mode == settling_mode
        sync_phase_mask = hold_mask_now | settling_mask_now
        self.phase_right = torch.where(sync_phase_mask, self.phase, phase_right_default)
        self.leg_phase = torch.stack([self.phase_left, self.phase_right], dim=1)
        # 统一落点规划：在每步相位更新后执行，供 clearance/alternation 共享
        self._update_foothold_planner()
        # Play/Test 调试：可视化 planner 预测落点（四角矩形）
        if self.cfg.env.test and self.viewer is not None:
            self._debug_draw_planner_targets(env_idx=0)

        # Planner 触地跟随诊断：记录“预测落点 vs 实际落点”误差（独立栏目）
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
        if not hasattr(self, "planner_diag_prev_contact"):
            self.planner_diag_prev_contact = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        touchdown_event = contact_now & (~self.planner_diag_prev_contact)
        self.planner_diag_prev_contact = contact_now

        if hasattr(self, "foothold_plan_target_xy") and hasattr(self, "foothold_plan_target_z"):
            xy_err = torch.norm(self.feet_pos[:, :, :2] - self.foothold_plan_target_xy, dim=2)
            z_err = torch.abs(self.feet_pos[:, :, 2] - self.foothold_plan_target_z)
            plan_conf = self.foothold_plan_conf if hasattr(self, "foothold_plan_conf") else torch.ones_like(xy_err)
            cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05).unsqueeze(1)
            valid_touch = touchdown_event & (plan_conf > 0.25)

            # 台阶任务触地样本：改硬过滤为软权重，避免“无事件=无学习”
            stair_touch_soft = torch.ones_like(plan_conf, dtype=torch.float)
            if hasattr(self, "foothold_plan_edge_count") or hasattr(self, "foothold_plan_top_out"):
                stair_touch_mask = torch.zeros_like(valid_touch)
                if hasattr(self, "foothold_plan_edge_count"):
                    stair_touch_mask = stair_touch_mask | (self.foothold_plan_edge_count > 0.5)
                if hasattr(self, "foothold_plan_top_out"):
                    stair_touch_mask = stair_touch_mask | self.foothold_plan_top_out
                stair_touch_soft = 0.20 + 0.80 * stair_touch_mask.float()

            valid_touch_f = valid_touch.float() * stair_touch_soft * cmd_forward_soft
            touch_count_step = torch.sum(valid_touch_f)
            if touch_count_step > 0:
                xy_err_step_mean = torch.sum(xy_err * valid_touch_f) / touch_count_step
                z_err_step_mean = torch.sum(z_err * valid_touch_f) / touch_count_step
                xy_hit4_step = torch.sum((xy_err < 0.04).float() * valid_touch_f) / touch_count_step
                xy_hit_step = torch.sum((xy_err < 0.08).float() * valid_touch_f) / touch_count_step
            else:
                xy_err_step_mean = torch.tensor(0.0, device=self.device)
                z_err_step_mean = torch.tensor(0.0, device=self.device)
                xy_hit4_step = torch.tensor(0.0, device=self.device)
                xy_hit_step = torch.tensor(0.0, device=self.device)

            # 回合累计统计（每个环境分开累计，日志取全局平均）
            if hasattr(self, "planner_diag_touch_count"):
                self.planner_diag_touch_count += torch.sum(valid_touch_f, dim=1)
                self.planner_diag_touch_xy_err_sum += torch.sum(xy_err * valid_touch_f, dim=1)
                self.planner_diag_touch_z_err_sum += torch.sum(z_err * valid_touch_f, dim=1)
                step_xy_max_env = torch.max(torch.where(valid_touch, xy_err, torch.zeros_like(xy_err)), dim=1).values
                self.planner_diag_touch_xy_err_max = torch.maximum(self.planner_diag_touch_xy_err_max, step_xy_max_env)

                total_touch_cum = torch.sum(self.planner_diag_touch_count)
                if total_touch_cum > 0:
                    xy_err_cum_mean = torch.sum(self.planner_diag_touch_xy_err_sum) / total_touch_cum
                    z_err_cum_mean = torch.sum(self.planner_diag_touch_z_err_sum) / total_touch_cum
                    valid_env = self.planner_diag_touch_count > 0
                    if valid_env.any():
                        xy_err_cum_max = torch.max(self.planner_diag_touch_xy_err_max[valid_env])
                    else:
                        xy_err_cum_max = torch.tensor(0.0, device=self.device)
                else:
                    xy_err_cum_mean = torch.tensor(0.0, device=self.device)
                    z_err_cum_mean = torch.tensor(0.0, device=self.device)
                    xy_err_cum_max = torch.tensor(0.0, device=self.device)

                self.extras["Diagnostics/planner_touch_count_step"] = touch_count_step.item()
                self.extras["Diagnostics/planner_touch_xy_err_step_mean"] = xy_err_step_mean.item()
                self.extras["Diagnostics/planner_touch_z_err_step_mean"] = z_err_step_mean.item()
                self.extras["Diagnostics/planner_touch_xy_hit_rate_4cm_step"] = xy_hit4_step.item()
                self.extras["Diagnostics/planner_touch_xy_hit_rate_8cm_step"] = xy_hit_step.item()
                self.extras["Diagnostics/planner_touch_count_cum"] = total_touch_cum.item()
                self.extras["Diagnostics/planner_touch_xy_err_cum_mean"] = xy_err_cum_mean.item()
                self.extras["Diagnostics/planner_touch_z_err_cum_mean"] = z_err_cum_mean.item()
                self.extras["Diagnostics/planner_touch_xy_err_cum_max"] = xy_err_cum_max.item()
                if hasattr(self, "planner_xy_err_ema"):
                    if float(touch_count_step.item()) > 0.5:
                        ema_alpha = 0.06
                        self.planner_xy_err_ema = (1.0 - ema_alpha) * self.planner_xy_err_ema + \
                            ema_alpha * xy_err_step_mean.detach().unsqueeze(0)
                        self.planner_hit8_ema = (1.0 - ema_alpha) * self.planner_hit8_ema + \
                            ema_alpha * xy_hit_step.detach().unsqueeze(0)
                        self.planner_hit4_ema = (1.0 - ema_alpha) * self.planner_hit4_ema + \
                            ema_alpha * xy_hit4_step.detach().unsqueeze(0)
                    else:
                        # 无有效触地时回归“中性基线”，而不是衰减到“完美”
                        neutral_xy = torch.tensor(0.18, device=self.device)
                        neutral_hit8 = torch.tensor(0.18, device=self.device)
                        neutral_hit4 = torch.tensor(0.08, device=self.device)
                        if hasattr(self, "foothold_plan_edge_count") or hasattr(self, "foothold_plan_top_out"):
                            stair_task_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                            if hasattr(self, "foothold_plan_edge_count"):
                                stair_task_env = stair_task_env | (torch.max(self.foothold_plan_edge_count, dim=1).values > 0.5)
                            if hasattr(self, "foothold_plan_top_out"):
                                stair_task_env = stair_task_env | torch.any(self.foothold_plan_top_out, dim=1)
                            stair_task_rate = torch.mean(stair_task_env.float())
                            # 有楼梯任务却没有触地事件时，提高 miss 先验，促使调度抬 planner_tracking
                            neutral_xy = torch.where(stair_task_rate > 0.10, torch.tensor(0.24, device=self.device), neutral_xy)
                            neutral_hit8 = torch.where(stair_task_rate > 0.10, torch.tensor(0.10, device=self.device), neutral_hit8)
                            neutral_hit4 = torch.where(stair_task_rate > 0.10, torch.tensor(0.03, device=self.device), neutral_hit4)
                        ema_alpha_idle = 0.010
                        self.planner_xy_err_ema = (1.0 - ema_alpha_idle) * self.planner_xy_err_ema + \
                            ema_alpha_idle * neutral_xy.unsqueeze(0)
                        self.planner_hit8_ema = (1.0 - ema_alpha_idle) * self.planner_hit8_ema + \
                            ema_alpha_idle * neutral_hit8.unsqueeze(0)
                        self.planner_hit4_ema = (1.0 - ema_alpha_idle) * self.planner_hit4_ema + \
                            ema_alpha_idle * neutral_hit4.unsqueeze(0)

                    self.extras["Diagnostics/planner_xy_err_ema"] = float(self.planner_xy_err_ema.item())
                    self.extras["Diagnostics/planner_hit8_ema"] = float(self.planner_hit8_ema.item())
                    self.extras["Diagnostics/planner_hit4_ema"] = float(self.planner_hit4_ema.item())
        
        # # [新增] 记录落足瞬间的坐标
        # if not hasattr(self, 'foothold_log'):
        #     self.foothold_log = [] # 暂时存在内存里，或者你可以定期写入文件

        # # 检测从摆动到支撑的瞬间 (Touch down)
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # touchdown = contact & (~self.last_contacts) # 上一帧没踩，这一帧踩了
        
        # if touchdown.any():
        #     env_ids, foot_ids = torch.where(touchdown)
        #     for i in range(len(env_ids)):
        #         e_idx = env_ids[i]
        #         f_idx = foot_ids[i]
        #         # 记录此时脚的世界坐标 (相对于台阶中心的相对位置更好，这里先存绝对位置)
        #         pos = self.feet_pos[e_idx, f_idx].cpu().numpy()
        #         # 存入列表：[环境ID, 步数, 坐标x, 坐标y, 坐标z]
        #         self.foothold_log.append([e_idx.item(), self.common_step_counter, pos[0], pos[1], pos[2]])

        # return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ 
        组装学生 (808维) 和 老师 (998维) 的非对称观测向量 
        学生 = 150维历史本体感受 + 18维planner轨迹切片特征 + 640维视觉特征
        老师 = 学生观测 + 187维地形高度采样 + 3维物理特权参数 (摩擦、负载、真速)
        """
        # 1. 计算当前时间相位 (2 维: sin/cos)
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        # 零指令静止模式下冻结相位观测，避免策略被振荡相位持续“诱导迈步”
        no_cmd_obs = self._get_no_cmd_mask().unsqueeze(1)
        sin_phase = torch.where(no_cmd_obs, torch.zeros_like(sin_phase), sin_phase)
        cos_phase = torch.where(no_cmd_obs, torch.ones_like(cos_phase), cos_phase)

        # 2. 构造当前帧的本体感受数据 (50 维)
        # 顺序必须固定：线速度(3), 角速度(3), 重力投影(3), 指令(3), 关节位置(12), 关节速度(12), 上一次动作(12), 相位(2)
        cmd_obs = self.commands[:, :3] * self.commands_scale
        if no_cmd_obs.any():
            cmd_obs[no_cmd_obs.squeeze(1)] = 0.0

        current_proprio = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,           # 3
            self.base_ang_vel * self.obs_scales.ang_vel,           # 3
            self.projected_gravity,                                # 3
            cmd_obs,                                               # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel,                # 12
            self.actions,                                          # 12
            sin_phase,                                             # 1
            cos_phase                                              # 1
        ), dim=-1)
        # 零指令下屏蔽“上一拍动作”观测，减少 residual action 对静止策略的持续诱导
        if no_cmd_obs.any():
            current_proprio[no_cmd_obs.squeeze(1), 36:48] = 0.0

        # 3. 更新历史堆叠缓冲区 (50 * 3 = 150 维)
        # 如果开启噪声，仅对当前帧加噪，防止历史数据噪声累积
        if self.add_noise:
            current_proprio += (2 * torch.rand_like(current_proprio) - 1) * self.noise_scale_vec[:50]
        
        # 将新数据推入，舍弃最旧的一帧
        next_history = torch.cat((current_proprio, self.obs_history_buf[:, :-50]), dim=-1)
        # reset 后 warm-start 历史，避免“全零历史”触发策略瞬态抬腿/抖动
        if hasattr(self, "reset_obs_warmstart_mask"):
            warm_mask = self.reset_obs_warmstart_mask
            if warm_mask.any():
                next_history[warm_mask] = current_proprio[warm_mask].repeat(1, 3)
                self.reset_obs_warmstart_mask[warm_mask] = False
        self.obs_history_buf = next_history

        # 4. 处理并获取归一化视觉观测 (640 维)
        if hasattr(self, 'visual_obs_buf'):
            max_range = self.cfg.sensor.stereo_cam.max_range
            # 将深度图 Clamp 在量程内并做归一化处理
            visual_obs = torch.clamp(self.visual_obs_buf, 0.0, max_range) / max_range
        else:
            visual_obs = torch.zeros((self.num_envs, 640), device=self.device)

        # 5. Planner 轨迹切片特征 (18 维):
        #    [dx,dy(4), dz_goal(2), z_err_now(2), phase_t(2), time_left(2), dz_start(2), conf(2), active(2)]
        planner_obs = torch.zeros((self.num_envs, 18), device=self.device)
        if hasattr(self, "foothold_plan_target_xy") and hasattr(self, "feet_pos"):
            plan_delta_w = self.foothold_plan_target_xy - self.feet_pos[:, :, :2]  # (N,2,2)
            yaw = self.rpy[:, 2]
            cos_yaw = torch.cos(yaw).unsqueeze(1)
            sin_yaw = torch.sin(yaw).unsqueeze(1)
            dx_b = cos_yaw * plan_delta_w[:, :, 0] + sin_yaw * plan_delta_w[:, :, 1]
            dy_b = -sin_yaw * plan_delta_w[:, :, 0] + cos_yaw * plan_delta_w[:, :, 1]
            plan_delta_b = torch.stack([dx_b, dy_b], dim=-1)
            # 归一化到网络友好范围，避免极端落点误差瞬间主导观测
            planner_obs[:, :4] = torch.clamp(plan_delta_b, min=-0.80, max=0.80).reshape(self.num_envs, -1) * 1.25
            if hasattr(self, "foothold_plan_dz_goal"):
                planner_obs[:, 4:6] = torch.clamp(self.foothold_plan_dz_goal, min=-0.35, max=0.35) * 2.0
            if hasattr(self, "foothold_plan_target_z"):
                z_err_now = self.feet_pos[:, :, 2] - self.foothold_plan_target_z
                planner_obs[:, 6:8] = torch.clamp(z_err_now, min=-0.30, max=0.30) * 3.0
            swing_start = 0.55
            swing_duration_phase = 1.0 - swing_start
            phase_t = torch.clamp((self.leg_phase - swing_start) / swing_duration_phase, min=0.0, max=1.0)
            planner_obs[:, 8:10] = phase_t * 2.0 - 1.0
            time_left_norm = 1.0 - phase_t
            planner_obs[:, 10:12] = time_left_norm * 2.0 - 1.0
            if hasattr(self, "foothold_plan_dz_start"):
                planner_obs[:, 12:14] = torch.clamp(self.foothold_plan_dz_start, min=-0.35, max=0.35) * 2.0
            if hasattr(self, "foothold_plan_conf"):
                planner_obs[:, 14:16] = torch.clamp(self.foothold_plan_conf, min=0.0, max=1.0)
            if hasattr(self, "foothold_plan_active"):
                planner_obs[:, 16:18] = self.foothold_plan_active.float()
            if no_cmd_obs.any():
                planner_obs[no_cmd_obs.squeeze(1)] = 0.0

        # 6. 【组装学生观测】 (150 + 18 + 640 = 808 维)
        # 注意：保持最后 640 维是视觉输入，兼容 ActorCriticTS 的视觉切片逻辑。
        self.obs_buf = torch.cat((self.obs_history_buf, planner_obs, visual_obs), dim=-1)

        # 7. 【组装老师观测特权信息】
        # 7.1 精准地形采样 (187 维)
        h_scale = getattr(self.obs_scales, 'height_measure', 5.0)
        # 计算相对高度差 (基座高度 - 腿长 - 地形采样点高度)
        # 腿长约 0.75m (从URDF计算: hip_pitch到ankle_roll总长)
        LEG_LENGTH = 0.75
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - LEG_LENGTH - self.measured_heights, -1, 1.) * h_scale
        
        # 7.2 物理特权信息 (3 维)
        # 维度 0: 摩擦力系数 (解决 play.py 报错：使用 getattr 保底，并从 CPU 同步到 GPU)
        friction = getattr(self, 'friction_coeffs', torch.ones(self.num_envs, 1, device='cpu'))
        self.privileged_physics_params[:, 0] = friction.to(self.device).squeeze()
        
        # 维度 1: 负载增量质量 (来自拦截函数 _process_rigid_body_props 记录的数据)
        self.privileged_physics_params[:, 1] = self.payloads.to(self.device).squeeze()
        
        # 维度 2: 真实前进 X 速度 (上帝视角绝对真值)
        self.privileged_physics_params[:, 2] = self.base_lin_vel[:, 0] * self.obs_scales.lin_vel
        
        # 8. 【最终输出】 (808 + 187 + 3 = 998 维)
        self.privileged_obs_buf = torch.cat((self.obs_buf, heights, self.privileged_physics_params), dim=-1)

        # 返回值在基类 step 函数中会被截断处理
        return self.obs_buf
    def create_sim(self):
        """ Creates simulation, terrain and environments """
        # ✅ 先初始化Warp库
        wp.init()
        print("Warp库初始化成功")
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        # if mesh_type == 'trimesh':
        #     self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'trimesh':
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
        # ✅ 新增：初始化双目相机（在环境创建之后）
        self.init_warp_stereo_camera()

    def _create_trimesh(self):
        """ 
        在仿真环境中创建三角形网格地形，并根据配置参数设置其物理属性。
        Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        """
        # 初始化三角形网格参数对象（用于配置物理引擎中的网格地形）
        tm_params = gymapi.TriangleMeshParams()
        
        # 设置地形网格的顶点数量（从self.terrain.vertices中获取）
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        
        # 设置地形网格的三角形面片数量（从self.terrain.triangles中获取）
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        # 设置地形在X轴方向的偏移（向左移动border_size距离，确保地形边界对齐）
        tm_params.transform.p.x = -self.cfg.terrain.border_size 
        
        # 设置地形在Y轴方向的偏移（向后移动border_size距离）
        tm_params.transform.p.y = -self.cfg.terrain.border_size
        
        # 设置地形在Z轴方向的偏移（无垂直偏移）
        tm_params.transform.p.z = 0.0
        
        # 设置地形的静摩擦系数（从配置中读取）
        tm_params.static_friction = self.cfg.terrain.static_friction
        
        # 设置地形的动摩擦系数（从配置中读取）
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        
        # 设置地形的弹性恢复系数（从配置中读取）
        tm_params.restitution = self.cfg.terrain.restitution
        
        # 将三角形网格添加到仿真环境中：
        # - 顶点数据按C语言顺序展平（行优先）
        # - 三角形索引数据按C语言顺序展平
        # - 传入配置参数tm_params
        self.gym.add_triangle_mesh(
            self.sim, 
            self.terrain.vertices.flatten(order='C'), 
            self.terrain.triangles.flatten(order='C'), 
            tm_params
        )   
        
        # 将地形高度图数据转换为PyTorch张量，并调整为二维矩阵形式，最后转移到指定设备（如GPU）
        self.height_samples = torch.tensor(
            self.terrain.height_field_raw.copy(),  # 再次确保独立
            dtype=torch.float32
        ).to(self.device)
        self.height_map = torch.tensor(
            self.terrain.height_field_raw.copy(),  # 复制原始数据
            dtype=torch.float32
        ).to(self.device) * self.cfg.terrain.vertical_scale  # 乘以垂直缩放系数
        # 预计算梯度图（米/米）：运行时查表，避免每步邻域差分重复计算
        self.height_grad_x_map = torch.zeros_like(self.height_map, device=self.device, dtype=torch.float32)
        self.height_grad_y_map = torch.zeros_like(self.height_map, device=self.device, dtype=torch.float32)
        step = float(self.cfg.terrain.horizontal_scale)
        if self.height_map.shape[0] > 1:
            self.height_grad_x_map[1:-1, :] = (self.height_map[2:, :] - self.height_map[:-2, :]) / (2.0 * step)
            self.height_grad_x_map[0, :] = (self.height_map[1, :] - self.height_map[0, :]) / step
            self.height_grad_x_map[-1, :] = (self.height_map[-1, :] - self.height_map[-2, :]) / step
        if self.height_map.shape[1] > 1:
            self.height_grad_y_map[:, 1:-1] = (self.height_map[:, 2:] - self.height_map[:, :-2]) / (2.0 * step)
            self.height_grad_y_map[:, 0] = (self.height_map[:, 1] - self.height_map[:, 0]) / step
            self.height_grad_y_map[:, -1] = (self.height_map[:, -1] - self.height_map[:, -2]) / step

        # 预计算台阶边缘查表（由 Terrain 在初始化阶段构建）
        self.stair_edge_back_x_pos = None
        self.stair_edge_front_x_pos = None
        self.stair_edge_back_x_neg = None
        self.stair_edge_front_x_neg = None
        self.stair_edge_back_y_pos = None
        self.stair_edge_front_y_pos = None
        self.stair_edge_back_y_neg = None
        self.stair_edge_front_y_neg = None
        self.stair_edge_h_thr = float(getattr(self.cfg.terrain, "edge_height_threshold", 0.03))
        self.has_stair_edge_lookup = False

        if hasattr(self.terrain, "stair_edge_lookup"):
            lookup = self.terrain.stair_edge_lookup
            self.stair_edge_back_x_pos = torch.tensor(lookup["back_x_pos"], dtype=torch.int32, device=self.device)
            self.stair_edge_front_x_pos = torch.tensor(lookup["front_x_pos"], dtype=torch.int32, device=self.device)
            self.stair_edge_back_x_neg = torch.tensor(lookup["back_x_neg"], dtype=torch.int32, device=self.device)
            self.stair_edge_front_x_neg = torch.tensor(lookup["front_x_neg"], dtype=torch.int32, device=self.device)
            self.stair_edge_back_y_pos = torch.tensor(lookup["back_y_pos"], dtype=torch.int32, device=self.device)
            self.stair_edge_front_y_pos = torch.tensor(lookup["front_y_pos"], dtype=torch.int32, device=self.device)
            self.stair_edge_back_y_neg = torch.tensor(lookup["back_y_neg"], dtype=torch.int32, device=self.device)
            self.stair_edge_front_y_neg = torch.tensor(lookup["front_y_neg"], dtype=torch.int32, device=self.device)
            self.stair_edge_h_thr = float(getattr(self.terrain, "stair_edge_height_threshold", self.stair_edge_h_thr))
            self.has_stair_edge_lookup = True
    
    # 请放在 g1_env.py 的 G1Robot 类中
    def _resample_commands(self, env_ids):
        """
        指令重采样与软牵引课程：
        - 楼梯段：前进牵引 + 课程化零指令驻立样本；
        - 平地段：前进牵引 + 少量零指令 + 探索样本。
        """
        # 注意：不在“普通指令重采样”中清 planner/latch 缓存。
        # 缓存只在 reset_idx 中按回合边界清理，避免中途重采样导致规划闭环断裂。

        def _apply_zero_commands(target_env_ids):
            if target_env_ids.numel() == 0:
                return
            self.commands[target_env_ids, 0] = 0.0
            self.commands[target_env_ids, 1] = 0.0
            if self.cfg.commands.heading_command:
                forward = quat_apply(self.base_quat[target_env_ids], self.forward_vec[target_env_ids])
                current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[target_env_ids, 3] = current_heading
            else:
                self.commands[target_env_ids, 2] = 0.0
            if hasattr(self, "explore_cmd_mask"):
                self.explore_cmd_mask[target_env_ids] = False

        def _idle_start_eligible(target_env_ids):
            """
            仅在“当前状态足够稳”时注入 no_cmd，避免中途急切零命令把样本推向姿态超限。
            该门控只影响 no_cmd 子集，不改 move/stair 主任务轨迹。
            """
            if target_env_ids.numel() == 0:
                return torch.zeros(0, dtype=torch.bool, device=self.device)
            speed_thr = float(getattr(self.cfg.commands, "no_cmd_start_speed_thr", 0.45))
            yaw_thr = float(getattr(self.cfg.commands, "no_cmd_start_yaw_thr", 0.80))
            tilt_thr = float(getattr(self.cfg.commands, "no_cmd_start_tilt_thr", 0.50))
            speed_xy = torch.norm(self.base_lin_vel[target_env_ids, :2], dim=1)
            yaw_abs = torch.abs(self.base_ang_vel[target_env_ids, 2])
            roll_abs = torch.abs(self.rpy[target_env_ids, 0])
            pitch_abs = torch.abs(self.rpy[target_env_ids, 1])
            left_contact = self.contact_forces[target_env_ids, self.feet_indices[0], 2] > 1.2
            right_contact = self.contact_forces[target_env_ids, self.feet_indices[1], 2] > 1.2
            has_support = left_contact | right_contact
            return (
                (speed_xy < speed_thr)
                & (yaw_abs < yaw_thr)
                & (roll_abs < tilt_thr)
                & (pitch_abs < tilt_thr)
                & has_support
            )

        manual_cmd_override = bool(getattr(self.cfg.env, "manual_cmd_override", False))
        allow_test_resample = bool(getattr(self.cfg.env, "allow_test_resample", False))
        if self.cfg.env.test and (manual_cmd_override or (not allow_test_resample)):
            # Play/Test 默认不在 reset 时注入随机指令，避免“零指令场景偶发跳变”
            _apply_zero_commands(env_ids)
            return

        work_env_ids = env_ids
        if work_env_ids.numel() == 0:
            return

        # no_cmd 软 governor：只调采样概率，不做按环境数的硬配额回收，
        # 避免 no_cmd_rate 曲线出现“台阶/脉冲”。
        target_no_cmd = float(getattr(self.cfg.commands, "target_no_cmd_rate", 0.30))
        target_no_cmd_low = float(getattr(self.cfg.commands, "target_no_cmd_rate_low", 0.25))
        target_no_cmd_high = float(getattr(self.cfg.commands, "target_no_cmd_rate_high", 0.35))
        target_no_cmd = float(np.clip(target_no_cmd, target_no_cmd_low, target_no_cmd_high))

        if hasattr(self, "_get_no_cmd_mask"):
            try:
                no_cmd_rate_now = float(torch.mean(self._get_no_cmd_mask().float()).item())
            except TypeError:
                no_cmd_rate_now = float(torch.mean(self._get_no_cmd_mask(planar_thr=0.05, yaw_thr=0.05).float()).item())
        else:
            no_cmd_rate_now = 0.0

        # no_cmd 比例闭环防振：
        # 1) 先做 EMA 低通，避免“当前瞬时占比”直接驱动下一轮概率；
        # 2) 加死区，目标附近不再来回修正；
        # 3) 降低 bias 增益，减小过冲。
        if not hasattr(self, "no_cmd_rate_ema"):
            self.no_cmd_rate_ema = torch.tensor([no_cmd_rate_now], dtype=torch.float, device=self.device)
        ema_alpha = 0.10
        self.no_cmd_rate_ema = (1.0 - ema_alpha) * self.no_cmd_rate_ema + ema_alpha * no_cmd_rate_now
        no_cmd_rate_ctrl = float(self.no_cmd_rate_ema.item())

        no_cmd_err_raw = target_no_cmd - no_cmd_rate_ctrl
        err_deadband = 0.02
        if abs(no_cmd_err_raw) < err_deadband:
            no_cmd_err = 0.0
        else:
            no_cmd_err = no_cmd_err_raw
        no_cmd_err = float(np.clip(no_cmd_err, -0.12, 0.12))
        # 当 no_cmd 占比远高于目标时，提供更强的回拉力度，尽快把样本还给主任务。
        stair_idle_prob_bias = 0.26 * no_cmd_err
        flat_idle_prob_bias = 0.60 * no_cmd_err

        # 1. 基础采样：调用基类生成包含前后左右、自转的彻底随机全向指令
        super()._resample_commands(work_env_ids)
        
        # 默认标记为“非探索样本”，仅在探索分支显式打开
        if hasattr(self, "explore_cmd_mask"):
            self.explore_cmd_mask[work_env_ids] = False

        # 2. 【概率软牵引课程】 (仅在三维地形上生效)
        if self.cfg.terrain.mesh_type == 'trimesh':

            # 重采样时先刷新这批环境的高度，避免使用上一位置的旧测高
            if self.cfg.terrain.measure_heights and work_env_ids.numel() > 0:
                self.measured_heights[work_env_ids] = self._get_heights(work_env_ids)

            on_stairs_mask = self._estimate_on_stairs_mask(env_ids=work_env_ids, update_state=True)
            force_test_stairs_mode = bool(getattr(self.cfg.env, "force_test_stairs_mode", False))
            if self.cfg.env.test and force_test_stairs_mode:
                on_stairs_mask = torch.ones_like(on_stairs_mask, dtype=torch.bool)

            # 关键分流修正：
            # 使用“脚下平坦度”判定是否应进入楼梯牵引，避免平台中心/台阶前缘被误分到楼梯分支，
            # 导致零指令静止样本被长期稀释。
            if hasattr(self, "measured_heights"):
                _, _, _, _, front_delta_all, underfoot_range_all, local_range_all = self._get_height_roi_features()
                underfoot_flat_mask = underfoot_range_all[work_env_ids] < 0.022
                front_jump_mask = front_delta_all[work_env_ids] > 0.010
                local_rough_mask = local_range_all[work_env_ids] > 0.035
                stair_conf_local = self._estimate_on_stairs_confidence(env_ids=work_env_ids, update_state=False)
                stair_conf_strong = stair_conf_local > 0.34
            else:
                underfoot_flat_mask = ~on_stairs_mask
                front_jump_mask = torch.zeros_like(on_stairs_mask, dtype=torch.bool)
                local_rough_mask = torch.zeros_like(on_stairs_mask, dtype=torch.bool)
                stair_conf_strong = torch.zeros_like(on_stairs_mask, dtype=torch.bool)

            # 【楼梯牵引】：
            # - 默认：楼梯且脚下非平坦；
            # - 兜底：即便脚下平坦，只要前方跳变/局部粗糙/楼梯置信强，也保持楼梯牵引。
            traction_mask = on_stairs_mask & (
                (~underfoot_flat_mask) | front_jump_mask | local_rough_mask | stair_conf_strong
            )

            if traction_mask.any():
                traction_env_ids = work_env_ids[traction_mask]
                stair_idle_prob_base = float(getattr(self.cfg.commands, "stair_idle_prob", 0.03))
                stair_idle_prob_early = float(
                    getattr(self.cfg.commands, "stair_idle_prob_early", max(stair_idle_prob_base, 0.12))
                )
                stair_idle_prob_late = float(
                    getattr(self.cfg.commands, "stair_idle_prob_late", stair_idle_prob_base)
                )
                stair_idle_prob_early = max(0.0, min(stair_idle_prob_early, 0.45))
                stair_idle_prob_late = max(0.0, min(stair_idle_prob_late, 0.45))
                if hasattr(self, "current_terrain_progress"):
                    prog_all = self.current_terrain_progress
                    if not torch.is_tensor(prog_all):
                        prog_all = torch.full((self.num_envs,), float(prog_all), device=self.device)
                    elif prog_all.ndim == 0:
                        prog_all = prog_all.repeat(self.num_envs)
                    else:
                        prog_all = prog_all.to(self.device)
                    traction_prog = torch.clamp(prog_all[traction_env_ids], min=0.0, max=1.0)
                else:
                    traction_prog = torch.zeros(len(traction_env_ids), device=self.device)
                stair_idle_prob_env = stair_idle_prob_late + \
                    (stair_idle_prob_early - stair_idle_prob_late) * (1.0 - traction_prog)
                stair_idle_prob_env = stair_idle_prob_env + stair_idle_prob_bias
                stair_idle_prob_env = torch.clamp(stair_idle_prob_env, min=0.0, max=0.20)

                # 楼梯 no_cmd：概率采样 + 上限截断（保底静止能力，不挤占主任务样本）。
                stair_idle_cap_ratio = float(getattr(self.cfg.commands, "stair_idle_hard_cap_ratio", 0.05))
                stair_idle_cap_ratio = float(np.clip(stair_idle_cap_ratio, 0.0, 0.25))
                stair_idle_cap = int(round(stair_idle_cap_ratio * len(traction_env_ids)))
                idle_mask_stair = torch.rand(len(traction_env_ids), device=self.device) < stair_idle_prob_env
                idle_count = int(torch.sum(idle_mask_stair.float()).item())
                if idle_count > stair_idle_cap:
                    idle_indices = torch.where(idle_mask_stair)[0]
                    keep_perm = torch.randperm(idle_indices.numel(), device=self.device)
                    keep_ids = idle_indices[keep_perm[:stair_idle_cap]]
                    new_mask = torch.zeros_like(idle_mask_stair)
                    new_mask[keep_ids] = True
                    idle_mask_stair = new_mask
                active_mask_stair = ~idle_mask_stair

                if idle_mask_stair.any():
                    idle_indices = torch.where(idle_mask_stair)[0]
                    idle_env_ids = traction_env_ids[idle_indices]
                    idle_eligible = _idle_start_eligible(idle_env_ids)
                    if idle_eligible.any():
                        _apply_zero_commands(idle_env_ids[idle_eligible])
                    idle_not_ready = ~idle_eligible
                    if idle_not_ready.any():
                        active_mask_stair[idle_indices[idle_not_ready]] = True

                if active_mask_stair.any():
                    active_env_ids = traction_env_ids[active_mask_stair]
                    # a. 楼梯推进速度：按轴向对齐/角点置信做温和调制，避免冲向直角边
                    stair_min_forward = float(getattr(self.cfg.commands, "stair_min_forward", 0.30))
                    max_forward = float(self.command_ranges["lin_vel_x"][1])
                    stair_min_forward = min(stair_min_forward, max_forward)

                    if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
                        stair_axis, _, corner_conf_all, _ = self._estimate_stair_axis_state()
                        yaw_t = self.rpy[active_env_ids, 2]
                        body_fwd_t = torch.stack([torch.cos(yaw_t), torch.sin(yaw_t)], dim=1)
                        axis_align_t = torch.abs(torch.sum(body_fwd_t * stair_axis[active_env_ids], dim=1))
                        corner_t = corner_conf_all[active_env_ids]

                        # 对齐越好速度上限越高；角点处进一步降速，给策略留出“先转正再上”时间
                        align_speed = 0.72 + 0.28 * axis_align_t
                        corner_slow = 1.0 - 0.45 * corner_t
                        speed_scale = torch.clamp(align_speed * corner_slow, min=0.55, max=1.0)

                        min_cmd = torch.clamp(stair_min_forward * speed_scale, min=0.14, max=max_forward - 0.05)
                        max_cmd = torch.clamp(max_forward * (0.80 + 0.20 * axis_align_t), min=0.24, max=max_forward)
                        max_cmd = torch.maximum(max_cmd, min_cmd + 0.03)

                        rand_u = torch.rand(len(active_env_ids), device=self.device)
                        self.commands[active_env_ids, 0] = min_cmd + (max_cmd - min_cmd) * rand_u

                        # b. 收紧侧移（按轴向对齐自适应），避免楼梯段形成“斜着上”
                        lat_scale = torch.clamp(0.04 + 0.05 * axis_align_t, min=0.03, max=0.09)
                        self.commands[active_env_ids, 1] *= lat_scale
                    else:
                        self.commands[active_env_ids, 0] = torch_rand_float(
                            stair_min_forward, max_forward,
                            (active_mask_stair.sum().item(), 1), device=self.device
                        ).squeeze(1)
                        # b. 收紧侧移，防止在爬楼梯时走出夸张的“螃蟹步”
                        self.commands[active_env_ids, 1] *= 0.1

                    # c. 处理旋转指令
                    if self.cfg.commands.heading_command:
                        forward = quat_apply(self.base_quat[active_env_ids], self.forward_vec[active_env_ids])
                        current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                        self.commands[active_env_ids, 3] = current_heading
                    else:
                        self.commands[active_env_ids, 2] = 0.0

            # 【平地上：前向牵引 + 少量零命令 + 全向自由探索】
            flat_mask = ~traction_mask
            if flat_mask.any():
                flat_env_ids = work_env_ids[flat_mask]
                
                # 读取分布概率
                flat_idle_prob_base = float(getattr(self.cfg.commands, "flat_idle_prob", 0.12))
                # 自适应静止样本：仅在“零指令稳定性差”时临时提高，不长期抬高配比
                if hasattr(self, "no_cmd_instability_ema"):
                    instab = float(torch.clamp(self.no_cmd_instability_ema, min=0.0, max=1.6).item())
                else:
                    instab = 0.0
                instab_boost = max(0.0, min((instab - 0.18) / 0.82, 1.0))
                # instability 提升幅度减半，避免与 no_cmd governor 叠加引发占比振荡。
                flat_idle_prob = flat_idle_prob_base + 0.015 * instab_boost + flat_idle_prob_bias
                flat_idle_prob = float(np.clip(flat_idle_prob, 0.0, 0.20))
                flat_forward_prob = float(getattr(self.cfg.commands, "flat_forward_prob", 0.78))
                flat_yaw_range = abs(float(getattr(self.cfg.commands, "flat_yaw_range", 0.1)))

                # flat no_cmd：概率采样（去掉硬配额回收，减小脉冲）。
                idle_mask_flat = torch.rand(len(flat_env_ids), device=self.device) < flat_idle_prob
                active_mask_flat = ~idle_mask_flat

                # 先处理零命令样本：仅对“可平稳切换”的子集注入 no_cmd
                if idle_mask_flat.any():
                    idle_indices = torch.where(idle_mask_flat)[0]
                    idle_env_ids = flat_env_ids[idle_indices]
                    idle_eligible = _idle_start_eligible(idle_env_ids)
                    if idle_eligible.any():
                        _apply_zero_commands(idle_env_ids[idle_eligible])
                    idle_not_ready = ~idle_eligible
                    if idle_not_ready.any():
                        active_mask_flat[idle_indices[idle_not_ready]] = True

                # 有指令活动的机器人
                if active_mask_flat.any():
                    active_env_ids = flat_env_ids[active_mask_flat]
                    active_rand = torch.rand(len(active_env_ids), device=self.device)
                    
                    forward_mask_flat = active_rand < flat_forward_prob
                    explore_mask_flat = ~forward_mask_flat

                    # 1. 前向牵引分支 (主流任务)
                    if forward_mask_flat.any():
                        forward_env_ids = active_env_ids[forward_mask_flat]
                        min_forward = min(0.25, float(self.command_ranges["lin_vel_x"][1]))
                        self.commands[forward_env_ids, 0] = torch_rand_float(
                            min_forward, self.command_ranges["lin_vel_x"][1],
                            (forward_mask_flat.sum().item(), 1), device=self.device
                        ).squeeze(1)
                        # 前向走的时候依然收紧侧移
                        self.commands[forward_env_ids, 1] *= 0.05
                        if self.cfg.commands.heading_command:
                            forward = quat_apply(self.base_quat[forward_env_ids], self.forward_vec[forward_env_ids])
                            current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                            self.commands[forward_env_ids, 3] = current_heading
                        else:
                            self.commands[forward_env_ids, 2] = torch_rand_float(
                                -flat_yaw_range, flat_yaw_range,
                                (len(forward_env_ids), 1), device=self.device
                            ).squeeze(1)

                    # 2. 全向自由探索分支 (那 10% 的环境)
                    if explore_mask_flat.any():
                        explore_env_ids = active_env_ids[explore_mask_flat]
                        
                        # 【核心改动】：
                        # 1. 不修改 commands[:, 0]，保留基类生成的 [-1.0, 1.0] 的随机前后速度
                        # 2. 不压缩 commands[:, 1]，保留基类生成的 [-0.5, 0.5] 的随机横移速度
                        
                        if not self.cfg.commands.heading_command:
                            # 稍微放开一点旋转限制，但不要转太快防止摔倒
                            explore_yaw = min(2.0 * flat_yaw_range, 0.4)
                            self.commands[explore_env_ids, 2] = torch_rand_float(
                                -explore_yaw, explore_yaw,
                                (len(explore_env_ids), 1), device=self.device
                            ).squeeze(1)
                            
                        # 标记为探索样本，免除视觉老司机的纠偏，并给探索任务适当降权
                        if hasattr(self, "explore_cmd_mask"):
                            self.explore_cmd_mask[explore_env_ids] = True

    def _debug_visualize_training(self, depth_tensor):
        try:
            # --- 1. 数据准备 (保持原始数值不被破坏) ---
            max_r = getattr(self.cfg.sensor.stereo_cam, 'max_range', 3.0)
            
            if isinstance(depth_tensor, torch.Tensor):
                # 使用 detach() 确保不影响计算图
                raw_map = depth_tensor.detach().cpu().numpy()
            else:
                raw_map = depth_tensor
                
            if raw_map.ndim == 3: 
                raw_map = raw_map.squeeze(0)
                
            H, W = raw_map.shape  # 20, 32

            # --- 2. 基础底图渲染 ---
            # 归一化并进行 Gamma 校正，让深色区域（近处）对比度更高
            norm_data = np.clip(raw_map / max_r, 0, 1)
            gamma_data = np.power(norm_data, 1.5) 
            depth_gray = (gamma_data * 255).astype(np.uint8)
            
            # 32x20 放大到 640x400，使用 INTER_NEAREST 保证像素块边界清晰
            upscale_factor = 20
            display_h, display_w = H * upscale_factor, W * upscale_factor
            display_img = cv2.resize(depth_gray, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
            
            # 转为 BGR 方便后续可能的彩色标注
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            # --- 3. 辅助标记：无效值点 ---
            # 遍历原始低分辨率地图，寻找无效点并在放大后的图上标绿点
            for y in range(H):
                for x in range(W):
                    if raw_map[y, x] < 0.01: # 假设小于 1cm 为无效值
                        # 计算放大后该像素块的中心坐标
                        cx = x * upscale_factor + (upscale_factor // 2)
                        cy = y * upscale_factor + (upscale_factor // 2)
                        cv2.circle(display_img, (cx, cy), 2, (0, 255, 0), -1)

            # --- 4. 实时信息显示 ---
            min_v, max_v = np.min(raw_map), np.max(raw_map)
            cv2.putText(display_img, f"Raw Depth: {min_v:.2f}-{max_v:.2f}m", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # --- 5. 窗口输出 ---
            cv2.imshow("Robot Depth Observation", display_img)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Visualization Error: {e}")

    def _debug_draw_planner_targets(self, env_idx=0):
        """
        在 viewer 中绘制 planner 预测落足矩形（按四角点几何）：
        - 每只脚一个目标矩形（前12cm、后5cm、半宽3cm）；
        - 颜色区分左右脚；
        - 额外绘制“当前脚中心 -> 目标中心”连线，便于观察跟踪误差。
        """
        if self.viewer is None:
            return
        if (not hasattr(self, "foothold_plan_target_xy")) or (not hasattr(self, "feet_num")):
            return
        if env_idx < 0 or env_idx >= self.num_envs:
            return

        # 每步刷新，避免历史线条残留
        self.gym.clear_lines(self.viewer)

        front_len = 0.12
        back_len = 0.05
        half_width = 0.03
        z_offset = 0.01

        feet_num = int(min(self.feet_num, self.foothold_plan_target_xy.shape[1]))
        if feet_num <= 0:
            return

        target_xy = self.foothold_plan_target_xy[env_idx, :feet_num]
        if hasattr(self, "foothold_plan_target_z"):
            target_z = self.foothold_plan_target_z[env_idx, :feet_num]
        else:
            target_z = self._get_terrain_heights(
                None, target_xy[:, 0], target_xy[:, 1], interpolate=False
            )

        if hasattr(self, "foothold_plan_active"):
            plan_active = self.foothold_plan_active[env_idx, :feet_num]
        else:
            plan_active = torch.ones(feet_num, dtype=torch.bool, device=self.device)

        if hasattr(self, "foothold_plan_conf"):
            plan_conf = self.foothold_plan_conf[env_idx, :feet_num]
        else:
            plan_conf = torch.ones(feet_num, dtype=torch.float, device=self.device)

        if hasattr(self, "foothold_plan_top_out"):
            top_out = self.foothold_plan_top_out[env_idx, :feet_num]
        else:
            top_out = torch.zeros(feet_num, dtype=torch.bool, device=self.device)

        # 仅绘制“当前处于规划任务”或“置信度足够”的脚，减少画面噪声
        draw_mask = plan_active | (plan_conf > 0.35)

        # Play 可视化：矩形朝向使用“楼梯轴向（带符号）”，避免瞬时侧向速度导致的斜矩形观感。
        yaw = self.rpy[env_idx, 2]
        body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=0)
        if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
            stair_axis, _, _, _ = self._estimate_stair_axis_state()
            axis = stair_axis[env_idx]
            cmd_axis_proj = torch.dot(self.commands[env_idx, :2], axis)
            body_axis_proj = torch.dot(body_fwd, axis)
            axis_sign_hint = cmd_axis_proj if torch.abs(cmd_axis_proj) > 0.03 else body_axis_proj
            axis_sign = torch.sign(axis_sign_hint)
            if float(torch.abs(axis_sign).item()) < 1e-5:
                axis_sign = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            fwd_unit = axis * axis_sign
        else:
            fwd_unit = body_fwd / (torch.norm(body_fwd) + 1e-6)
        foot_fwd = fwd_unit.view(1, 2).repeat(feet_num, 1)
        foot_lat = torch.stack([-foot_fwd[:, 1], foot_fwd[:, 0]], dim=-1)

        verts = []
        colors = []

        for i in range(feet_num):
            if not bool(draw_mask[i].item()):
                continue

            c_xy = target_xy[i]
            z = float(target_z[i].item() + z_offset)
            fwd = foot_fwd[i]
            lat = foot_lat[i]

            c0 = c_xy + fwd * front_len + lat * half_width
            c1 = c_xy + fwd * front_len - lat * half_width
            c2 = c_xy - fwd * back_len - lat * half_width
            c3 = c_xy - fwd * back_len + lat * half_width

            corners = torch.stack([c0, c1, c2, c3], dim=0)
            corners3 = torch.zeros((4, 3), device=self.device, dtype=torch.float32)
            corners3[:, :2] = corners
            corners3[:, 2] = z

            if bool(top_out[i].item()):
                # 登顶过渡样本用紫色
                color_box = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            elif i == 0:
                # 左脚：青色
                color_box = np.array([0.0, 1.0, 1.0], dtype=np.float32)
            else:
                # 右脚：橙色
                color_box = np.array([1.0, 0.6, 0.0], dtype=np.float32)

            edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            for s, e in edges:
                verts.append(corners3[s].detach().cpu().numpy())
                verts.append(corners3[e].detach().cpu().numpy())
                colors.append(color_box)

            # 朝向线（矩形后中点 -> 前中点）
            rear_mid = c_xy - fwd * back_len
            front_mid = c_xy + fwd * front_len
            rear_pt = torch.zeros(3, device=self.device, dtype=torch.float32)
            rear_pt[:2] = rear_mid
            rear_pt[2] = z
            front_pt = torch.zeros(3, device=self.device, dtype=torch.float32)
            front_pt[:2] = front_mid
            front_pt[2] = z
            verts.append(rear_pt.detach().cpu().numpy())
            verts.append(front_pt.detach().cpu().numpy())
            colors.append(color_box)

            # 当前脚中心到目标中心的误差连线（黄色）
            cur_center = self.feet_pos[env_idx, i, :3]
            tgt_center = torch.zeros(3, device=self.device, dtype=torch.float32)
            tgt_center[:2] = c_xy
            tgt_center[2] = z
            verts.append(cur_center.detach().cpu().numpy())
            verts.append(tgt_center.detach().cpu().numpy())
            colors.append(np.array([1.0, 1.0, 0.0], dtype=np.float32))

        if len(colors) == 0:
            return

        verts_np = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        colors_np = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        self.gym.add_lines(self.viewer, self.envs[env_idx], len(colors), verts_np, colors_np)

    # 添加到 G1Robot 类中，最好放在 _update_warp_camera 附近
    def _debug_draw_camera_rays(self, env_idx, cam_global_pos, cam_global_rot):
        """ [修复版V3] 自动适配数量(防崩溃) + RGB坐标轴 """
        if self.viewer is None: return

        self.gym.clear_lines(self.viewer)

        # --- 1. 绘制黄色视锥 ---
        from isaacgym.torch_utils import quat_apply
        import torch.nn.functional as F

        # [关键] 动态获取数量，不要写死 5
        num_rays = self.debug_rays_local.shape[0]

        rot_expanded = cam_global_rot.unsqueeze(0).repeat(num_rays, 1)
        rotated_rays = quat_apply(rot_expanded, self.debug_rays_local)
        rotated_rays = F.normalize(rotated_rays, p=2, dim=1) * 3.0 
        
        global_ends = cam_global_pos.unsqueeze(0) + rotated_rays
        starts_np = cam_global_pos.unsqueeze(0).repeat(num_rays, 1).cpu().numpy()
        ends_np = global_ends.cpu().numpy()
        
        # 加粗
        base_verts = np.stack([starts_np, ends_np], axis=1).reshape(-1, 3) 
        offset = 0.002
        thick_verts = np.vstack([
            base_verts, base_verts + np.array([offset, offset, 0.0]),
            base_verts - np.array([offset, offset, 0.0]), base_verts + np.array([-offset, offset, 0.0])
        ]).astype(np.float32) # [关键] 强制 float32
        
        colors_yellow = np.vstack([self.debug_rays_colors] * 4).astype(np.float32)

        self.gym.add_lines(self.viewer, self.envs[env_idx], num_rays * 4, thick_verts, colors_yellow)

        # --- 2. 绘制 RGB 坐标轴 (红X 绿Y 蓝Z) ---
        axis_len = 0.5
        local_axes = torch.tensor([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]], device=self.device)
        
        rot_axes = cam_global_rot.unsqueeze(0).repeat(3, 1)
        global_axes = quat_apply(rot_axes, local_axes)
        
        starts_ax = cam_global_pos.unsqueeze(0).repeat(3, 1)
        ends_ax = starts_ax + global_axes
        verts_ax = torch.stack([starts_ax, ends_ax], dim=1).reshape(-1, 3).cpu().numpy()
        
        colors_rgb = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)

        thick_verts_ax = np.vstack([
            verts_ax, verts_ax + np.array([offset, offset, 0.0]),
            verts_ax - np.array([offset, offset, 0.0])
        ]).astype(np.float32)
        
        thick_colors_rgb = np.vstack([colors_rgb] * 3).astype(np.float32)

        self.gym.add_lines(self.viewer, self.envs[env_idx], 9, thick_verts_ax, thick_colors_rgb)

    def _get_terrain_heights(self, env_ids, x, y, interpolate=True):
        """
        [修复版] 根据世界坐标 (x, y) 获取地形高度
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros_like(x)
        
        # 1. 安全检查
        if not hasattr(self, 'height_samples'):
            return torch.zeros_like(x)
            
        num_rows, num_cols = self.height_samples.shape
        
        horizontal_scale = self.cfg.terrain.horizontal_scale
        vertical_scale = self.cfg.terrain.vertical_scale
        border = self.cfg.terrain.border_size

        # 3. 转换坐标（连续网格坐标）
        x_grid = (x.to(torch.float32) + border) / horizontal_scale
        y_grid = (y.to(torch.float32) + border) / horizontal_scale

        # 4. 可选插值模式
        x_grid = torch.clamp(x_grid, 0.0, float(num_rows - 1))
        y_grid = torch.clamp(y_grid, 0.0, float(num_cols - 1))

        # 边缘判定等几何任务可关闭插值，保留台阶立边的“硬边”特性
        if not interpolate:
            x_idx = torch.floor(x_grid).long()
            y_idx = torch.floor(y_grid).long()
            heights = self.height_samples[x_idx, y_idx].to(torch.float32)
            return heights * vertical_scale

        # 双线性插值（比整格取整更平滑，减少 10cm 量化抖动）
        x0 = torch.floor(x_grid).long()
        y0 = torch.floor(y_grid).long()
        x1 = torch.clamp(x0 + 1, max=num_rows - 1)
        y1 = torch.clamp(y0 + 1, max=num_cols - 1)

        wx = x_grid - x0.to(torch.float32)
        wy = y_grid - y0.to(torch.float32)

        h00 = self.height_samples[x0, y0].to(torch.float32)
        h10 = self.height_samples[x1, y0].to(torch.float32)
        h01 = self.height_samples[x0, y1].to(torch.float32)
        h11 = self.height_samples[x1, y1].to(torch.float32)

        h0 = h00 * (1.0 - wx) + h10 * wx
        h1 = h01 * (1.0 - wx) + h11 * wx
        heights = h0 * (1.0 - wy) + h1 * wy
        return heights * vertical_scale

    def _lookup_grid_map(self, map_tensor, x, y, interpolate=True):
        """
        通用网格查表：输入世界坐标 (x, y)，查询任意二维标量场 map_tensor。
        map_tensor 采用与 height_samples 相同网格坐标系（米制值）。
        """
        if map_tensor is None or self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros_like(x, dtype=torch.float32)

        num_rows, num_cols = map_tensor.shape
        step = float(self.cfg.terrain.horizontal_scale)
        border = float(self.cfg.terrain.border_size)

        x_grid = torch.clamp((x.to(torch.float32) + border) / step, 0.0, float(num_rows - 1))
        y_grid = torch.clamp((y.to(torch.float32) + border) / step, 0.0, float(num_cols - 1))

        if not interpolate:
            x_idx = torch.floor(x_grid).long()
            y_idx = torch.floor(y_grid).long()
            return map_tensor[x_idx, y_idx].to(torch.float32)

        x0 = torch.floor(x_grid).long()
        y0 = torch.floor(y_grid).long()
        x1 = torch.clamp(x0 + 1, max=num_rows - 1)
        y1 = torch.clamp(y0 + 1, max=num_cols - 1)
        wx = x_grid - x0.to(torch.float32)
        wy = y_grid - y0.to(torch.float32)

        v00 = map_tensor[x0, y0].to(torch.float32)
        v10 = map_tensor[x1, y0].to(torch.float32)
        v01 = map_tensor[x0, y1].to(torch.float32)
        v11 = map_tensor[x1, y1].to(torch.float32)

        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        return v0 * (1.0 - wy) + v1 * wy

    def _update_contact_metrics(self, hanging_mask, contact_mask):
        """ 统计全贴合率指标 """
        with torch.no_grad():
            if not hasattr(self, "extras"):
                self.extras = {}
            # 强制转为 Bool 进行位运算
            contact_bool = contact_mask[:, :, 0].bool()
            hanging_any_bool = hanging_mask.any(dim=-1).bool()
            
            # 全贴合判定：脚在地上 且 四个角都没悬空
            is_full_contact = contact_bool & (~hanging_any_bool)
            
            total_contacts = torch.sum(contact_bool).float()
            if total_contacts > 0:
                self.extras["metrics/full_contact_rate"] = torch.sum(is_full_contact).float() / total_contacts
            else:
                self.extras["metrics/full_contact_rate"] = torch.tensor(0.0, device=self.device)

    def _update_full_contact_rate_metric(self):
        """每步更新脚掌全贴合率（与奖励函数解耦，始终可观测）。"""
        with torch.no_grad():
            front_len = 0.12
            back_len = 0.05
            half_width = 0.03
            edge_threshold = float(getattr(self.cfg.rewards, "full_contact_edge_threshold", 0.01))

            if (not hasattr(self, "_full_contact_corners_local")) or \
               (self._full_contact_corners_local.device != self.device):
                self._full_contact_corners_local = torch.tensor(
                    [
                        [front_len,  half_width, 0.0],
                        [front_len, -half_width, 0.0],
                        [-back_len,  half_width, 0.0],
                        [-back_len, -half_width, 0.0],
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

            feet_pos = self.feet_pos
            feet_rot = self.rigid_body_states_view[:, self.feet_indices, 3:7]
            num_corners = self._full_contact_corners_local.shape[0]

            feet_rot_exp = feet_rot.unsqueeze(2).repeat(1, 1, num_corners, 1)
            corners_local_exp = self._full_contact_corners_local.view(1, 1, num_corners, 3).expand(
                self.num_envs, self.feet_num, num_corners, 3
            )
            corners_world = quat_apply(
                feet_rot_exp.reshape(-1, 4),
                corners_local_exp.reshape(-1, 3)
            ).view(self.num_envs, self.feet_num, num_corners, 3)
            corners_world += feet_pos.unsqueeze(2)

            terrain_h = self._get_terrain_heights(
                None,
                corners_world[..., 0].reshape(-1),
                corners_world[..., 1].reshape(-1),
                interpolate=False
            ).view(self.num_envs, self.feet_num, num_corners)

            height_diff = corners_world[..., 2] - terrain_h
            hanging_mask = height_diff > edge_threshold
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
            contact_mask = contact.unsqueeze(2).repeat(1, 1, num_corners)
            self._update_contact_metrics(hanging_mask, contact_mask)
    
    def reset_idx(self, env_ids):
        """ 重置环境时的钩子函数 """
        # 如果是在测试模式（play.py），且日志积累到一定量，则保存
        # 这里的 self.cfg.env.test 取决于你运行 play.py 时是否传入了相应参数
        # if len(self.foothold_log) > 100000: 
        #     self.save_foothold_data()
        self.obs_history_buf[env_ids] = 0
        if hasattr(self, "reset_obs_warmstart_mask"):
            self.reset_obs_warmstart_mask[env_ids] = True
        # [新增修复]：清除死亡机器人的跨步高度前世记忆
        if hasattr(self, 'cached_start_h'):
            self.cached_start_h[env_ids] = 0.0
            self.cached_end_h[env_ids] = 0.0
        if hasattr(self, 'cached_edge_obstacle_h'):
            self.cached_edge_obstacle_h[env_ids] = 0.0
        # 兼容旧缓存字段（历史 checkpoint）
        if hasattr(self, 'cached_path_peak_h'):
            self.cached_path_peak_h[env_ids] = 0.0
        if hasattr(self, 'on_stairs_state'):
            self.on_stairs_state[env_ids] = False
        if hasattr(self, 'on_stairs_conf_state'):
            self.on_stairs_conf_state[env_ids] = 0.0
        if hasattr(self, "foothold_plan_active"):
            self.foothold_plan_active[env_ids] = False
        if hasattr(self, "foothold_plan_ttl_steps"):
            self.foothold_plan_ttl_steps[env_ids] = 0
        if hasattr(self, "foothold_plan_start_xy"):
            self.foothold_plan_start_xy[env_ids] = 0.0
        if hasattr(self, "foothold_plan_target_xy"):
            self.foothold_plan_target_xy[env_ids] = 0.0
        if hasattr(self, "foothold_plan_target_z"):
            self.foothold_plan_target_z[env_ids] = 0.0
        if hasattr(self, "foothold_plan_start_h"):
            self.foothold_plan_start_h[env_ids] = 0.0
        if hasattr(self, "foothold_plan_riser_h"):
            self.foothold_plan_riser_h[env_ids] = 0.0
        if hasattr(self, "foothold_plan_edge_count"):
            self.foothold_plan_edge_count[env_ids] = 0.0
        if hasattr(self, "foothold_plan_dz_goal"):
            self.foothold_plan_dz_goal[env_ids] = 0.0
        if hasattr(self, "foothold_plan_dz_start"):
            self.foothold_plan_dz_start[env_ids] = 0.0
        if hasattr(self, "foothold_plan_conf"):
            self.foothold_plan_conf[env_ids] = 0.0
        if hasattr(self, "foothold_plan_top_out"):
            self.foothold_plan_top_out[env_ids] = False
        if hasattr(self, "foothold_plan_new_event"):
            self.foothold_plan_new_event[env_ids] = False
        if hasattr(self, "foothold_prev_swing_mask"):
            self.foothold_prev_swing_mask[env_ids] = False
        if hasattr(self, "foothold_plan_prev_contact"):
            self.foothold_plan_prev_contact[env_ids] = False
        if hasattr(self, "alt_prev_contact_feet"):
            self.alt_prev_contact_feet[env_ids] = False
        if hasattr(self, "alt_prev_lead_sign"):
            self.alt_prev_lead_sign[env_ids] = 0.0
        if hasattr(self, "alt_same_lead_steps"):
            self.alt_same_lead_steps[env_ids] = 0.0
        if hasattr(self, "alt_prev_ds_active"):
            self.alt_prev_ds_active[env_ids] = 0.0
        if hasattr(self, "planner_diag_prev_contact"):
            self.planner_diag_prev_contact[env_ids] = False
        if hasattr(self, "plan_track_prev_contact"):
            self.plan_track_prev_contact[env_ids] = False
        if hasattr(self, "planner_diag_touch_count"):
            self.planner_diag_touch_count[env_ids] = 0.0
        if hasattr(self, "planner_diag_touch_xy_err_sum"):
            self.planner_diag_touch_xy_err_sum[env_ids] = 0.0
        if hasattr(self, "planner_diag_touch_z_err_sum"):
            self.planner_diag_touch_z_err_sum[env_ids] = 0.0
        if hasattr(self, "planner_diag_touch_xy_err_max"):
            self.planner_diag_touch_xy_err_max[env_ids] = 0.0
        # 必须调用父类的 reset_idx，否则机器人摔倒后不会重置！
        super().reset_idx(env_ids)
        if hasattr(self, "episode_stair_conf_sum"):
            self.episode_stair_conf_sum[env_ids] = 0.0
        if hasattr(self, "episode_stair_conf_max"):
            self.episode_stair_conf_max[env_ids] = 0.0
        if hasattr(self, "episode_nav_steps"):
            self.episode_nav_steps[env_ids] = 0
        if hasattr(self, "episode_abs_vx_sum"):
            self.episode_abs_vx_sum[env_ids] = 0.0
        if hasattr(self, "episode_abs_vy_sum"):
            self.episode_abs_vy_sum[env_ids] = 0.0
        if hasattr(self, "episode_abs_wz_sum"):
            self.episode_abs_wz_sum[env_ids] = 0.0
        if hasattr(self, "explore_cmd_mask"):
            self.explore_cmd_mask[env_ids] = False
        # 零指令样本重置后强制清零根速度，避免“无指令却带随机初速”污染静止分布
        no_cmd_after_reset = self._get_no_cmd_mask()[env_ids]
        if no_cmd_after_reset.any():
            no_cmd_env_ids = env_ids[no_cmd_after_reset]
            self.root_states[no_cmd_env_ids, 7:13] = 0.0
            env_ids_int32 = no_cmd_env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )
        # 相位状态机重置：按回合初始命令选择 RUN/HOLD，避免跨回合残留状态污染
        if hasattr(self, "phase"):
            self.phase[env_ids] = 0.0
        if hasattr(self, "phase_left"):
            self.phase_left[env_ids] = 0.0
        if hasattr(self, "phase_right"):
            self.phase_right[env_ids] = 0.5
        if hasattr(self, "leg_phase"):
            self.leg_phase[env_ids, 0] = 0.0
            self.leg_phase[env_ids, 1] = 0.5
        if hasattr(self, "phase_ds_stable_steps"):
            self.phase_ds_stable_steps[env_ids] = 0
        if hasattr(self, "phase_move_cmd_steps"):
            self.phase_move_cmd_steps[env_ids] = 0
        if hasattr(self, "no_cmd_lost_support_steps"):
            self.no_cmd_lost_support_steps[env_ids] = 0
        if hasattr(self, "phase_resume_ramp"):
            # reset 后统一从 0 开始爬升，避免有运动指令样本直接以高步频起跳导致早摔。
            self.phase_resume_ramp[env_ids] = 0.0
        if hasattr(self, "phase_mode"):
            no_cmd_post = self._get_no_cmd_mask()[env_ids]
            hold_mode = torch.full((len(env_ids),), 2, dtype=torch.long, device=self.device)
            run_mode = torch.zeros((len(env_ids),), dtype=torch.long, device=self.device)
            self.phase_mode[env_ids] = torch.where(no_cmd_post, hold_mode, run_mode)
            # no_cmd 回合起步用双支撑同相，避免 reset 后单腿先抬导致后仰摔倒。
            if hasattr(self, "phase_right"):
                self.phase_right[env_ids] = torch.where(
                    no_cmd_post,
                    torch.zeros_like(self.phase_right[env_ids]),
                    torch.full_like(self.phase_right[env_ids], 0.5),
                )
            if hasattr(self, "leg_phase"):
                self.leg_phase[env_ids, 1] = torch.where(
                    no_cmd_post,
                    torch.zeros_like(self.leg_phase[env_ids, 1]),
                    torch.full_like(self.leg_phase[env_ids, 1], 0.5),
                )
            if hasattr(self, "phase_resume_ramp"):
                self.phase_resume_ramp[env_ids] = torch.where(
                    no_cmd_post,
                    torch.zeros_like(self.phase_resume_ramp[env_ids]),
                    torch.zeros_like(self.phase_resume_ramp[env_ids])
                )

    def save_foothold_data(self):
        """ 将记录的落足点保存为 CSV """
        import os
        # 确保 LEGGED_GYM_ROOT_DIR 已导入，如果没有，可以在函数内临时获取
        from legged_gym import LEGGED_GYM_ROOT_DIR
        
        if not self.foothold_log:
            return
            
        df = pd.DataFrame(self.foothold_log, columns=['env_id', 'step', 'x', 'y', 'z'])
        
        log_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, 
            'logs', 
            self.cfg.asset.name, 
            'data', 
            self.session_time # ✅ 每次运行都会有一个独立的文件夹
        )
        os.makedirs(log_dir, exist_ok=True)
        
        file_path = os.path.join(log_dir, f"footholds_iter_{self.common_step_counter}.csv")
        df.to_csv(file_path, index=False)
        
        # ！！！关键：清空列表，释放显存/内存
        self.foothold_log = [] 
        print(f"\033[92m[Data Export]\033[0m 已保存 {len(df)} 条落足点数据至: {file_path}")

    def _reward_approach_stairs(self):
        """
        [前置引导奖励]
        基于“当前课程等级对应的目标台阶高度”做前置引导。
        当前方高度差越接近目标台阶高度，且机器人有前进意图并实际前进时，奖励越大。
        """
        if not hasattr(self, 'measured_heights'):
            return torch.zeros(self.num_envs, device=self.device)

        center, front_near, front_far, _, _, _, _ = self._get_height_roi_features()
        front = torch.maximum(front_near, front_far)

        # 前方相对脚下的正向高度差（仅关心“前高后低”的上台阶趋势）
        delta_h = torch.clamp(front - center, min=0.0, max=0.25)

        # 与 terrain.py / stair_alternation 保持口径一致：difficulty = level / num_rows
        rows = float(max(self.cfg.terrain.num_rows, 1))
        difficulty = self.terrain_levels.float() / rows
        target_step_height = 0.05 + 0.1 * difficulty

        # 高度匹配奖励：越接近目标台阶高度越高（避免“只要有高度差就刷分”）
        # 放宽匹配带宽，减少平台中心/远台阶阶段的奖励稀疏
        sigma = 0.06
        height_match = torch.exp(-torch.square(delta_h - target_step_height) / (2.0 * sigma * sigma))
        near_stair = torch.sigmoid((delta_h - 0.010) / 0.005)

        progress_vel, _, _, _, stair_conf, _, _ = self._get_progress_velocity_state()
        forward_cmd_conf = torch.sigmoid((self.commands[:, 0] - 0.05) / 0.06)
        lateral_task_conf = torch.sigmoid((0.14 - torch.abs(self.commands[:, 1])) / 0.05)
        task_mode_conf = forward_cmd_conf * lateral_task_conf
        progress_conf = torch.sigmoid((progress_vel - 0.04) / 0.07)
        vel_gate = 0.08 + 0.92 * progress_conf

        # 该项只负责“靠近台阶”，进入台阶后快速淡出，避免与 first_step/rect/clearance 重叠。
        pre_stair_conf = torch.clamp((0.58 - stair_conf) / 0.38, min=0.0, max=1.0)
        pre_stair_gate = 0.02 + 0.98 * pre_stair_conf

        # 视觉可见台阶时增强牵引：解决“平台中心已看见台阶但牵引仍偏弱”
        if hasattr(self, "visual_obs_buf") and hasattr(self.cfg, "sensor") and self.cfg.sensor.stereo_cam.enable:
            cam_cfg = self.cfg.sensor.stereo_cam
            # 与 stair_alignment 一致，先做裁剪，降低无效深度值导致的假激活。
            depth_img = torch.clamp(
                self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width),
                0.1,
                2.0
            )
            h_start = int(cam_cfg.height * 0.55)
            w_split = max(1, cam_cfg.width // 3)
            depth_strip = torch.mean(depth_img[:, h_start:, :], dim=1)
            left_depth = torch.mean(depth_strip[:, :w_split], dim=1)
            center_depth = torch.mean(depth_strip[:, w_split:cam_cfg.width - w_split], dim=1)
            right_depth = torch.mean(depth_strip[:, cam_cfg.width - w_split:], dim=1)
            side_depth = 0.5 * (left_depth + right_depth)
            seen_depth_thr = float(getattr(cam_cfg, "vision_seen_depth_thresh", 1.25))
            seen_depth_sigma = max(float(getattr(cam_cfg, "vision_seen_depth_sigma", 0.18)), 1e-4)
            seen_edge_thr = float(getattr(cam_cfg, "vision_seen_edge_thresh", 0.10))
            seen_edge_sigma = max(float(getattr(cam_cfg, "vision_seen_edge_sigma", 0.03)), 1e-4)
            center_near_conf = torch.sigmoid((seen_depth_thr - center_depth) / seen_depth_sigma)
            center_step_conf = torch.sigmoid((torch.abs(side_depth - center_depth) - seen_edge_thr) / seen_edge_sigma)
            visual_seen_conf = torch.maximum(center_near_conf, center_step_conf)
        else:
            visual_seen_conf = torch.zeros(self.num_envs, device=self.device)
        seen_boost = 0.20 + 0.80 * visual_seen_conf

        # 零/弱前进指令时不再优化“靠近楼梯”，避免静止阶段被该项驱动去找台阶边
        cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05)
        task_gate = cmd_forward_soft * (0.02 + 0.98 * task_mode_conf)
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return height_match * near_stair * task_gate * vel_gate * pre_stair_gate * seen_boost * explore_gate

    def _reward_first_step_commit(self):
        """
        台阶前“第一步承诺奖励”：
        在临近楼梯且有前进意图时，鼓励出现“前足前探 + 双脚高度差接近目标台阶高”的动作，
        避免停在台阶前原地踏步。
        """
        if not hasattr(self, 'measured_heights'):
            return torch.zeros(self.num_envs, device=self.device)

        center, _, _, front, _, _, _ = self._get_height_roi_features()
        delta_h = torch.clamp(front - center, min=0.0, max=0.30)

        near_stair_conf = torch.sigmoid((delta_h - 0.012) / 0.006)
        progress_vel, _, v_lat, _, stair_conf, _, _ = self._get_progress_velocity_state()
        pre_stair_conf = torch.clamp((0.62 - stair_conf) / 0.22, min=0.0, max=1.0)
        pre_stair_gate = near_stair_conf * pre_stair_conf

        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.06)
        lateral_task_conf = torch.sigmoid((0.14 - torch.abs(self.commands[:, 1])) / 0.05)
        cmd_conf = cmd_forward_conf * lateral_task_conf
        progress_conf = torch.sigmoid((progress_vel - 0.04) / 0.07)
        progress_gate = 0.05 + 0.95 * progress_conf
        lat_motion_gate = torch.sigmoid((0.10 - torch.abs(v_lat)) / 0.04)

        rows = float(max(self.cfg.terrain.num_rows, 1))
        difficulty = self.terrain_levels.float() / rows
        target_step_height = 0.05 + 0.1 * difficulty
        z_diff = torch.abs(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2])
        z_match = torch.exp(-torch.square(z_diff - target_step_height) / (2.0 * 0.045 * 0.045))

        yaw = self.rpy[:, 2]
        root_x = self.root_states[:, 0].unsqueeze(1)
        root_y = self.root_states[:, 1].unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        foot_dx_w = self.feet_pos[:, :, 0] - root_x
        foot_dy_w = self.feet_pos[:, :, 1] - root_y
        foot_x_body = cos_yaw * foot_dx_w + sin_yaw * foot_dy_w
        lead_reach = torch.sigmoid((torch.max(foot_x_body, dim=1).values - 0.22) / 0.04)
        foot_x_span = torch.abs(foot_x_body[:, 0] - foot_x_body[:, 1])
        span_conf = torch.sigmoid((foot_x_span - 0.07) / 0.03)
        step_reach = lead_reach * (0.20 + 0.80 * span_conf)

        swing_phase_conf = torch.max(torch.sigmoid((self.leg_phase - 0.56) / 0.07), dim=1).values
        swing_vel_conf = torch.clamp(torch.max(torch.abs(self.feet_vel[:, :, 2]), dim=1).values / 0.20, min=0.0, max=1.0)
        motion_gate = 0.05 + 0.95 * (0.60 * swing_phase_conf + 0.40 * swing_vel_conf)

        commit_signal = 0.65 * z_match + 0.35 * step_reach

        # 零/弱前进指令时仅保留软门控，避免静止阶段被隐式牵引且避免梯度断流
        cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05)
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return pre_stair_gate * cmd_conf * progress_gate * lat_motion_gate * motion_gate * commit_signal * explore_gate * cmd_forward_soft
    
    def _reward_stair_alternation(self):
        """
        楼梯交替约束（Planner 解耦版）：
        1) 连续项：摆动全过程跟踪 planner 的 z 差目标（不再稀疏）；
        2) 事件项：触地瞬间检查 z 差是否落到 planner 目标；
        3) 结算项：双支撑事件只比较“是否换脚领先”，推进交替节奏。
        4) XY 落点监督完全交给 planner_tracking，本函数不再重复计算 XY。
        """
        if not hasattr(self, 'measured_heights'):
            if hasattr(self, "extras"):
                self.extras["Diagnostics/alt_dz_score_step"] = 0.0
                self.extras["Diagnostics/alt_switch_score_step"] = 0.0
                self.extras["Diagnostics/alt_switch_event_rate_step"] = 0.0
            return torch.zeros(self.num_envs, device=self.device)
        has_plan_cache = hasattr(self, "foothold_plan_conf") and bool((self.foothold_plan_conf > 0.05).any().item())
        if hasattr(self, "foothold_plan_active") and (not bool(self.foothold_plan_active.any().item())) and (not has_plan_cache):
            contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
            if hasattr(self, "alt_prev_contact_feet"):
                self.alt_prev_contact_feet = contact_now
            if hasattr(self, "alt_prev_ds_active"):
                self.alt_prev_ds_active = torch.zeros_like(self.alt_prev_ds_active)
            if hasattr(self, "extras"):
                self.extras["Diagnostics/alt_dz_score_step"] = 0.0
                self.extras["Diagnostics/alt_switch_score_step"] = 0.0
                self.extras["Diagnostics/alt_switch_event_rate_step"] = 0.0
            return torch.zeros(self.num_envs, device=self.device)

        plan_env_active = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_active"):
            plan_env_active = torch.any(self.foothold_plan_active, dim=1).float()
        if hasattr(self, "foothold_plan_conf"):
            plan_env_active = torch.maximum(
                plan_env_active,
                torch.clamp(torch.max(self.foothold_plan_conf, dim=1).values / 0.25, min=0.0, max=1.0)
            )
        plan_env_gate = 0.20 + 0.80 * plan_env_active

        # 楼梯 + 任务门控（平地/零指令静音）
        _, _, _, _, front_delta, underfoot_range, _ = self._get_height_roi_features()
        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        local_under_conf = torch.sigmoid((underfoot_range - 0.036) / 0.008)
        local_front_conf = torch.sigmoid((front_delta - 0.010) / 0.004)
        stair_gate = torch.clamp(torch.maximum(stair_conf, torch.maximum(local_under_conf, local_front_conf)), min=0.0, max=1.0)
        cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05)
        progress_vel, _, _, _, _, _, _ = self._get_progress_velocity_state()
        progress_conf = torch.sigmoid((progress_vel - 0.04) / 0.06)

        # Planner 目标：dz_start -> dz_goal（若 planner 未激活则回退到课程目标）
        step_h = torch.clamp(self._get_step_height_target(), min=0.05, max=0.20)
        task_gate_base = stair_gate * cmd_forward_soft
        late_prog = getattr(self, "current_late_stage_progress", 0.0)
        if torch.is_tensor(late_prog):
            late_scalar = torch.clamp(late_prog, min=0.0, max=1.0)
            if late_scalar.ndim == 0:
                late_scalar = late_scalar.repeat(self.num_envs)
            else:
                late_scalar = late_scalar.to(self.device)
        else:
            late_scalar = torch.full((self.num_envs,), float(np.clip(late_prog, 0.0, 1.0)), device=self.device)
        # 前期放松推进门控底座，保证 alternation 在“刚学会迈步”阶段有连续梯度；
        # 后期再逐步收紧，避免低速碎步刷分。
        alt_progress_floor = 0.22 + 0.18 * (1.0 - late_scalar)  # early~0.40, late~0.22
        progress_gate = alt_progress_floor + (1.0 - alt_progress_floor) * progress_conf
        # 硬静音：仅在“规划器确认跨边/登顶过渡”时参与打分，平地完全不算 alternation。
        hard_task_conf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_edge_count"):
            plan_edge_env = torch.max(self.foothold_plan_edge_count, dim=1).values
            hard_task_conf = torch.maximum(hard_task_conf, torch.clamp(plan_edge_env / 0.8, min=0.0, max=1.0))
        if hasattr(self, "foothold_plan_top_out"):
            hard_task_conf = torch.maximum(
                hard_task_conf,
                torch.any(self.foothold_plan_top_out, dim=1).float()
            )
        # 兜底：无 planner 缓存时，退回楼梯置信门控，避免函数完全失效
        if (not hasattr(self, "foothold_plan_edge_count")) and (not hasattr(self, "foothold_plan_top_out")):
            hard_task_conf = torch.clamp((stair_gate - 0.20) / 0.50, min=0.0, max=1.0)
        else:
            # 即便还没明确检测到跨边，也保留小比例楼梯软置信参与，避免前期完全断梯度。
            stair_soft_conf = torch.clamp((stair_gate - 0.22) / 0.50, min=0.0, max=1.0)
            hard_task_conf = torch.maximum(hard_task_conf, 0.35 * stair_soft_conf)
        # 早期提高监督底座，后期回落；避免“必须完美命中才有 alternation 学习信号”。
        alt_gate_floor = 0.16 + 0.14 * (1.0 - late_scalar)  # early~0.30, late~0.16
        task_gate = task_gate_base * (alt_gate_floor + (1.0 - alt_gate_floor) * hard_task_conf)
        task_gate = task_gate * plan_env_gate
        fallback_goal = step_h.unsqueeze(1).repeat(1, self.feet_num)
        fallback_start = -0.40 * fallback_goal
        dz_goal = self.foothold_plan_dz_goal if hasattr(self, "foothold_plan_dz_goal") else fallback_goal
        dz_start = self.foothold_plan_dz_start if hasattr(self, "foothold_plan_dz_start") else fallback_start
        plan_conf = self.foothold_plan_conf if hasattr(self, "foothold_plan_conf") else torch.ones_like(dz_goal)

        phase_t = torch.clamp((self.leg_phase - 0.55) / 0.45, min=0.0, max=1.0)
        swing_mask = (self.leg_phase > 0.55)
        swing_plan_w = torch.ones((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_active"):
            swing_plan_w = 0.20 + 0.80 * self.foothold_plan_active.float()

        # 每只脚相对另一只脚的实时 z 差
        left_z = self.feet_pos[:, 0, 2]
        right_z = self.feet_pos[:, 1, 2]
        dz_curr = torch.stack([left_z - right_z, right_z - left_z], dim=1)

        # 连续项：摆动全过程按相位跟踪 dz 轨迹
        dz_traj = dz_start + (dz_goal - dz_start) * phase_t
        dz_err = torch.abs(dz_curr - dz_traj)
        prog_sigma = 0.055
        prog_term = torch.exp(-torch.square(dz_err) / (2.0 * prog_sigma * prog_sigma)) \
            * swing_mask.float() * swing_plan_w * (0.35 + 0.65 * plan_conf)
        swing_count = torch.clamp(torch.sum(swing_mask.float() * swing_plan_w, dim=1), min=1.0)
        prog_score = torch.sum(prog_term, dim=1) / swing_count

        # 触地事件：落地瞬间 z 差要接近 planner 目标 dz
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
        touchdown_quality, _ = self._get_touchdown_contact_quality(contact_now)
        if not hasattr(self, "alt_prev_contact_feet"):
            self.alt_prev_contact_feet = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        touchdown_event = contact_now & (~self.alt_prev_contact_feet) & (self.leg_phase < 0.55) & touchdown_quality
        touchdown_w = touchdown_event.float()
        touch_sigma = 0.045
        touch_err = torch.abs(dz_curr - dz_goal)
        touch_term = torch.exp(-torch.square(touch_err) / (2.0 * touch_sigma * touch_sigma)) \
            * touchdown_w * (0.40 + 0.60 * plan_conf)
        touch_score = torch.clamp(torch.sum(touch_term, dim=1), min=0.0, max=1.2)

        # 解耦：alternation 只负责节奏/高度，不再承担 XY 落点监督（由 planner_tracking 独占）。
        self.alt_prev_contact_feet = contact_now

        # 双支撑切换事件：上一双支撑领先脚是否发生切换
        if not hasattr(self, "alt_prev_lead_sign"):
            self.alt_prev_lead_sign = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if not hasattr(self, "alt_prev_ds_active"):
            self.alt_prev_ds_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        double_support = (contact_now[:, 0] & contact_now[:, 1]) & (self.leg_phase[:, 0] < 0.55) & (self.leg_phase[:, 1] < 0.55)
        ds_event = double_support & (~(self.alt_prev_ds_active > 0.5))
        signed_z_diff = left_z - right_z
        lead_thr = torch.clamp(0.35 * step_h, min=0.02, max=0.07)
        ds_sign = torch.where(
            signed_z_diff > lead_thr,
            torch.ones_like(signed_z_diff),
            torch.where(signed_z_diff < -lead_thr, -torch.ones_like(signed_z_diff), torch.zeros_like(signed_z_diff))
        )
        prev_sign = self.alt_prev_lead_sign
        switch_bonus = ds_event & (torch.abs(prev_sign) > 0.5) & (ds_sign * prev_sign < 0.0)
        same_pen = ds_event & (torch.abs(prev_sign) > 0.5) & (ds_sign == prev_sign)
        amb_pen = ds_event & (ds_sign == 0.0)
        first_valid = ds_event & (torch.abs(prev_sign) <= 0.5) & (ds_sign != 0.0)
        # 未切换惩罚在低推进时加重：专治“低速跟随/原地等一下再跨”
        stall_conf = torch.clamp((0.10 - progress_vel) / 0.10, min=0.0, max=1.0)
        no_switch_gain = 1.0 + 0.8 * stall_conf
        switch_score = (
            0.78 * switch_bonus.float()
            + 0.12 * first_valid.float()
            - 1.10 * same_pen.float() * no_switch_gain
            - 0.42 * amb_pen.float() * no_switch_gain
        )
        switch_score = switch_score * plan_env_gate

        inactive = (task_gate < 0.08)
        next_sign = torch.where(ds_event & (ds_sign != 0.0), ds_sign, prev_sign)
        self.alt_prev_lead_sign = torch.where(inactive, torch.zeros_like(next_sign), next_sign)
        self.alt_prev_ds_active = torch.where(inactive, torch.zeros_like(self.alt_prev_ds_active), double_support.float())

        # 合成：仅使用 dz 连续项 + dz 触地项 + 换脚事件项
        dz_score = 0.40 * prog_score + 0.46 * touch_score
        raw_signal = dz_score + switch_score
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        alt_gate = task_gate * progress_gate * explore_gate

        # 诊断拆分：用于判断“高度跟踪没学好”还是“切换节奏没学好”
        if hasattr(self, "extras"):
            self.extras["Diagnostics/alt_dz_score_step"] = float(torch.mean(dz_score * alt_gate).item())
            self.extras["Diagnostics/alt_switch_score_step"] = float(torch.mean(switch_score * alt_gate).item())
            self.extras["Diagnostics/alt_switch_event_rate_step"] = float(torch.mean(ds_event.float() * plan_env_gate).item())

        return torch.clamp(raw_signal * task_gate * progress_gate, min=-1.6, max=1.4) * explore_gate

    def _reward_planner_tracking(self):
        """
        Planner 跟踪奖励（单职责，XY-only）：
        1) 摆动期连续项：鼓励脚的 XY 逐步贴近 planner 目标点；
        2) 触地事件项：课程化阈值（前期宽松，后期收紧到 4cm 级）；
        3) 仅楼梯任务 + 前进命令启用；平地/零命令静音；
        4) Z 方向由 stair_clearance 单独负责，本函数不约束 Z。
        """
        if not hasattr(self, "foothold_plan_target_xy"):
            return torch.zeros(self.num_envs, device=self.device)
        has_plan_cache = hasattr(self, "foothold_plan_conf") and bool((self.foothold_plan_conf > 0.05).any().item())
        if hasattr(self, "foothold_plan_active") and (not bool(self.foothold_plan_active.any().item())) and (not has_plan_cache):
            contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
            if hasattr(self, "plan_track_prev_contact"):
                self.plan_track_prev_contact = contact_now
            return torch.zeros(self.num_envs, device=self.device)

        plan_env_active = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_active"):
            plan_env_active = torch.any(self.foothold_plan_active, dim=1).float()
        if hasattr(self, "foothold_plan_conf"):
            plan_env_active = torch.maximum(
                plan_env_active,
                torch.clamp(torch.max(self.foothold_plan_conf, dim=1).values / 0.25, min=0.0, max=1.0)
            )
        plan_env_gate = 0.20 + 0.80 * plan_env_active

        cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05)

        # 楼梯任务硬门控：优先以 planner 的“跨边/登顶过渡”定义任务场景
        planner_stair_conf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_edge_count"):
            plan_edge_env = torch.max(self.foothold_plan_edge_count, dim=1).values
            planner_stair_conf = torch.maximum(
                planner_stair_conf,
                torch.clamp(plan_edge_env / 0.8, min=0.0, max=1.0)
            )
        if hasattr(self, "foothold_plan_top_out"):
            planner_stair_conf = torch.maximum(
                planner_stair_conf,
                torch.any(self.foothold_plan_top_out, dim=1).float()
            )
        if hasattr(self, "foothold_plan_edge_count") or hasattr(self, "foothold_plan_top_out"):
            stair_hard_gate = planner_stair_conf
        else:
            if hasattr(self, "measured_heights"):
                stair_conf = self._estimate_on_stairs_confidence(update_state=False)
                stair_hard_gate = torch.clamp((stair_conf - 0.15) / 0.50, min=0.0, max=1.0)
            else:
                stair_hard_gate = torch.ones(self.num_envs, device=self.device)

        # 软门控：保留不断流，但降低低速可得分空间，防止“跟点小碎步”局部最优。
        late_prog = getattr(self, "current_late_stage_progress", 0.0)
        if torch.is_tensor(late_prog):
            late_scalar = torch.clamp(late_prog, min=0.0, max=1.0)
            if late_scalar.ndim == 0:
                late_scalar = late_scalar.repeat(self.num_envs)
            else:
                late_scalar = late_scalar.to(self.device)
        else:
            late_scalar = torch.full((self.num_envs,), float(np.clip(late_prog, 0.0, 1.0)), device=self.device)
        # 前期放松 planner 门控底座，保证触地/摆动跟踪有可学习信号；
        # 后期再回落，防止非任务区刷分。
        planner_gate_floor = 0.18 + 0.16 * (1.0 - late_scalar)  # early~0.34, late~0.18
        task_gate = cmd_forward_soft * (planner_gate_floor + (1.0 - planner_gate_floor) * stair_hard_gate)
        no_cmd_mask = self._get_no_cmd_mask()
        task_gate = torch.where(no_cmd_mask, torch.zeros_like(task_gate), task_gate)
        progress_vel, _, _, _, _, _, _ = self._get_progress_velocity_state()
        progress_conf = torch.sigmoid((progress_vel - 0.05) / 0.05)
        plan_progress_floor = 0.35 + 0.20 * (1.0 - late_scalar)  # early~0.55, late~0.35
        task_gate = task_gate * (plan_progress_floor + (1.0 - plan_progress_floor) * progress_conf)
        task_gate = task_gate * plan_env_gate

        plan_conf = self.foothold_plan_conf if hasattr(self, "foothold_plan_conf") else \
            torch.ones((self.num_envs, self.feet_num), device=self.device)
        xy_err = torch.norm(self.feet_pos[:, :, :2] - self.foothold_plan_target_xy, dim=2)
        # 摆动轨迹跟踪统一放到 stair_clearance 内做（Z 主导 + 轻量 XY），
        # planner_tracking 只保留终点落足跟踪，避免双重监督冲突。
        xy_err_swing = xy_err
        # 单点可支撑评分（CReF 思路的轻量版）：
        # 仅在 planner 目标点上评估“可支撑性”，不做候选点搜索，计算开销小。
        support_conf = torch.ones((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
            plan_xy = self.foothold_plan_target_xy
            plan_x = plan_xy[:, :, 0].reshape(-1)
            plan_y = plan_xy[:, :, 1].reshape(-1)
            gx = self._lookup_grid_map(self.height_grad_x_map, plan_x, plan_y, interpolate=True).view(self.num_envs, self.feet_num)
            gy = self._lookup_grid_map(self.height_grad_y_map, plan_x, plan_y, interpolate=True).view(self.num_envs, self.feet_num)
            g_abs_x = torch.abs(gx)
            g_abs_y = torch.abs(gy)
            slope_mag = torch.sqrt(torch.clamp(g_abs_x * g_abs_x + g_abs_y * g_abs_y, min=0.0))
            slope_score = torch.exp(-torch.square(slope_mag / 0.28))
            g_min = torch.minimum(g_abs_x, g_abs_y)
            anis = torch.abs(g_abs_x - g_abs_y)
            corner_risk = torch.sigmoid((g_min - 0.010) / 0.003) * torch.sigmoid((0.006 - anis) / 0.002)
            support_conf = torch.clamp(
                0.20 + 0.80 * slope_score * (1.0 - 0.55 * corner_risk),
                min=0.12,
                max=1.0
            )

            terrain_z_at_plan = self._get_terrain_heights(None, plan_x, plan_y, interpolate=True).view(self.num_envs, self.feet_num)
            if hasattr(self, "foothold_plan_target_z"):
                z_ref = self.foothold_plan_target_z
            else:
                z_ref = terrain_z_at_plan
            z_match = torch.exp(-torch.square(z_ref - terrain_z_at_plan) / (2.0 * 0.018 * 0.018))
            support_conf = torch.clamp(support_conf * (0.35 + 0.65 * z_match), min=0.10, max=1.0)
        support_w = 0.30 + 0.70 * support_conf

        # 1) 摆动期连续跟踪
        swing_mask = self.leg_phase > 0.55
        swing_plan_w = torch.ones((self.num_envs, self.feet_num), dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_active"):
            swing_plan_w = 0.20 + 0.80 * self.foothold_plan_active.float()
        # 课程化：前期更宽松，后期收紧，减少早期因高误差长期负分而“学不动”
        late_prog = getattr(self, "current_late_stage_progress", 0.0)
        if torch.is_tensor(late_prog):
            late = torch.clamp(late_prog, min=0.0, max=1.0).unsqueeze(1)
        else:
            late = torch.full((self.num_envs, 1), float(late_prog), device=self.device)
        swing_sigma = 0.14 - 0.05 * late
        swing_track = torch.exp(-torch.square(xy_err_swing) / (2.0 * swing_sigma * swing_sigma)) \
            * swing_mask.float() * swing_plan_w * (0.35 + 0.65 * plan_conf) * support_w
        swing_count = torch.clamp(torch.sum(swing_mask.float() * swing_plan_w, dim=1), min=1.0)
        swing_score = torch.sum(swing_track, dim=1) / swing_count

        # 2) 触地事件结算：仅在“有效 planner 任务”统计命中，避免无效样本稀释梯度
        contact_now = self.contact_forces[:, self.feet_indices, 2] > 1.2
        touchdown_quality, _ = self._get_touchdown_contact_quality(contact_now)
        if not hasattr(self, "plan_track_prev_contact"):
            self.plan_track_prev_contact = torch.zeros((self.num_envs, self.feet_num), dtype=torch.bool, device=self.device)
        touchdown_raw = contact_now & (~self.plan_track_prev_contact) & (self.leg_phase < 0.55)
        touchdown_event = touchdown_raw & touchdown_quality
        side_hit_event = touchdown_raw & (~touchdown_quality)
        self.plan_track_prev_contact = contact_now
        effective_cmd_mask = (cmd_forward_soft > 0.35)
        touchdown_event = touchdown_event & effective_cmd_mask.unsqueeze(1)
        side_hit_event = side_hit_event & effective_cmd_mask.unsqueeze(1)
        touchdown_w = touchdown_event.float()

        hit_thr = 0.12 - 0.07 * late      # 12cm -> 5cm
        pen_thr = 0.20 - 0.10 * late      # 20cm -> 10cm
        # 触地事件软权重：不再“硬要求 stair_hard_gate>0.5 才统计”
        touch_task_w = (0.20 + 0.80 * stair_hard_gate).unsqueeze(1)
        touch_pen_w = 0.45 + 0.55 * support_conf
        penalty_soft_gate = (0.30 + 0.70 * progress_conf).unsqueeze(1)
        touch_reward = torch.clamp((hit_thr - xy_err) / torch.clamp(hit_thr, min=1e-4), min=0.0, max=1.0) \
            * touchdown_w * touch_task_w * (0.35 + 0.65 * plan_conf) * support_w
        touch_penalty = torch.clamp((xy_err - pen_thr) / 0.16, min=0.0, max=1.6) \
            * touchdown_w * touch_task_w * (0.35 + 0.65 * plan_conf) * touch_pen_w * penalty_soft_gate
        side_hit_penalty = side_hit_event.float() * touch_task_w * penalty_soft_gate
        support_touch_bonus = torch.clamp(
            torch.sum(touchdown_w * touch_task_w * support_conf, dim=1),
            min=0.0,
            max=1.2
        )

        touch_score = torch.clamp(torch.sum(touch_reward, dim=1), min=0.0, max=1.2)
        touch_pen_score = torch.clamp(torch.sum(touch_penalty, dim=1), min=0.0, max=1.6)
        side_hit_pen_score = torch.clamp(torch.sum(side_hit_penalty, dim=1), min=0.0, max=1.8)

        # 前期轻罚、后期收紧，避免早期高误差阶段被大负项直接压死
        pen_gain = 0.10 + 0.90 * torch.clamp(late.squeeze(1), min=0.0, max=1.0)
        # 强化“触地命中预测点”主导，摆动连续项作为辅助。
        raw_signal = (
            0.24 * swing_score
            + 0.88 * touch_score
            + 0.22 * support_touch_bonus
            - 1.10 * pen_gain * touch_pen_score
            - 0.45 * pen_gain * side_hit_pen_score
        )
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return torch.clamp(raw_signal, min=-1.2, max=1.4) * task_gate * explore_gate

    def _reward_foot_flatness(self):
        """
        [物理加固版] 区分垂直支撑与侧向碰撞，防止 Stumble 期间误判
        """
        # 1. 获取脚的世界坐标系旋转 (Quat: x, y, z, w)
        feet_rot = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        
        # 2. 极致数学优化：直接提取旋转矩阵第三列计算 sin^2(theta)
        qx, qy, qz, qw = feet_rot[..., 0], feet_rot[..., 1], feet_rot[..., 2], feet_rot[..., 3]
        proj_x = 2.0 * (qx * qz + qy * qw)
        proj_y = 2.0 * (qy * qz - qx * qw)
        horizontal_deviation = proj_x**2 + proj_y**2
        
        # --- [核心修改 1: 智能触地判定] ---
        forces = self.contact_forces[:, self.feet_indices, :]
        force_z = forces[..., 2]
        force_xy = torch.norm(forces[..., :2], dim=-1)
        contact = force_z > 1.0
        
        # 只有当垂直力大于 10.0N，且垂直力大于水平力的 2 倍时，才认为是在“稳固支撑”
        # 这样即便脚踢到了台阶侧面(水平力大)，也不会触发放平惩罚
        is_firm_contact = (force_z > 10.0) & (force_z > 2.0 * force_xy)
        
        # --- [核心修改 2: 动态约束权重] ---
        # 计算一个支撑置信度系数 (0 ~ 1.0)
        # 只有真正踩下去了(50N 以上)，才进入 1.0 的强制放平状态
        contact_confidence = torch.clip(force_z / 50.0, 0.0, 1.0)
        
        # 3. 支撑状态掩码：处于支撑相位 (<0.55) 【或者】 确认踩实
        is_stance_phase = (self.leg_phase < 0.55).float()
        
        # 支撑相惩罚：极其严厉，且随受力增大而增强
        stance_penalty = horizontal_deviation * is_stance_phase * contact_confidence
        
        # 摆动期只在“预落地末段”给轻惩罚，避免中前段抬腿被过度压制
        is_pure_swing = (1.0 - is_stance_phase) * (~is_firm_contact).float()
        pre_land_swing = torch.clamp((self.leg_phase - 0.80) / 0.20, min=0.0, max=1.0) * is_pure_swing
        swing_penalty = torch.clamp(horizontal_deviation - 0.25, min=0.0) * (0.4 * pre_land_swing)

        # 触地窗口（摆动末段 + 早期支撑段）强化放平约束，抑制“脚尖先着地”。
        touchdown_window = torch.clamp(
            torch.clamp((self.leg_phase - 0.80) / 0.20, min=0.0, max=1.0) +
            torch.clamp((0.12 - self.leg_phase) / 0.12, min=0.0, max=1.0),
            min=0.0,
            max=1.0
        )
        touchdown_contact = contact.float() * touchdown_window
        touchdown_penalty = torch.clamp(horizontal_deviation - 0.08, min=0.0) * (0.85 * touchdown_contact)
        
        # 4. 零指令静止时，适度保留放平约束：
        # - 不再完全关闭摆动/触地窗口分支；
        # - 保留中等强度支撑约束，避免“脚尖上翘/后仰倒”。
        no_cmd_mask = self._get_no_cmd_mask().unsqueeze(1)
        stance_scale = torch.ones_like(stance_penalty)
        swing_penalty = torch.where(no_cmd_mask, 0.35 * swing_penalty, swing_penalty)
        touchdown_penalty = torch.where(no_cmd_mask, 0.90 * touchdown_penalty, touchdown_penalty)

        # 5. 软门控：全程保留脚掌姿态约束，楼梯段适度增强
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            flatness_gate = 0.7 + 0.4 * stair_conf  # [0.7, 1.1]
        else:
            flatness_gate = torch.ones(self.num_envs, device=self.device)

        return torch.sum(stance_penalty * stance_scale + swing_penalty + touchdown_penalty, dim=1) * flatness_gate

    def _reward_foot_support_rect(self):
        """
        以踝关节为核心的台阶支撑奖励（预计算边缘查表版）：
        1) 先做四角点高度一致性快筛，非候选脚不做重检测；
        2) 根据脚朝向确定前后轴（x/y 其一），按方向查最近 back/front 边缘；
        3) 计算“踝到后沿边缘线”的垂直距离 + “脚尖到前沿立边”净距；
        4) 按 5cm=0, 9cm=1, 13cm=0 打分，台阶外仍给负惩罚。
        """
        # 课程进度（用于阈值轻微收紧）
        progress = getattr(self, 'current_late_stage_progress',
                           getattr(self, 'current_terrain_progress', 0.0))
        if torch.is_tensor(progress):
            prog = progress.unsqueeze(1)
        else:
            prog = torch.full((self.num_envs, 1), float(progress), device=self.device)

        # 脚几何（来自你的约束）：前 12cm，后 5cm；台阶宽 30cm -> 踝可行域 13cm
        front_len = 0.12
        back_len = 0.05
        half_width = 0.03
        feasible_cap = 0.13
        feasible_lower = 0.05  # 新增：下限 5cm
        feasible_peak = 0.09

        feet_xy = self.feet_pos[:, :, :2]
        ankle_x = feet_xy[:, :, 0]
        ankle_y = feet_xy[:, :, 1]
        ankle_h = self._get_terrain_heights(
            None,
            ankle_x.reshape(-1),
            ankle_y.reshape(-1),
            interpolate=False
        ).view(self.num_envs, self.feet_num)

        # 前后轴使用“任务推进方向”（基座速度方向，低速回退到机体前向），
        # 抑制脚局部朝向导致的内八/斜切落脚。
        yaw = self.rpy[:, 2]
        base_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        world_vel = self.root_states[:, 7:9]
        vel_norm = torch.norm(world_vel, dim=1, keepdim=True)
        vel_dir = torch.where(vel_norm > 0.05, world_vel / (vel_norm + 1e-6), base_fwd)
        vel_forward_proj = torch.sum(vel_dir * base_fwd, dim=1, keepdim=True)
        cmd_forward_conf = torch.sigmoid((self.commands[:, 0].unsqueeze(1) - 0.06) / 0.04)
        backward_conf = cmd_forward_conf * torch.sigmoid((-vel_forward_proj - 0.02) / 0.06)
        task_fwd = (1.0 - backward_conf) * vel_dir + backward_conf * base_fwd
        task_fwd_norm = torch.norm(task_fwd, dim=1, keepdim=True)
        task_fwd = torch.where(task_fwd_norm > 1e-5, task_fwd / (task_fwd_norm + 1e-6), base_fwd)
        foot_fwd = task_fwd.unsqueeze(1).repeat(1, self.feet_num, 1)
        foot_lat = torch.stack([-foot_fwd[:, :, 1], foot_fwd[:, :, 0]], dim=-1)

        # 四角点：只用于“是否在台阶内”的软判定（不再要求四角齐平）
        off_fb = torch.tensor([front_len, front_len, -back_len, -back_len], device=self.device, dtype=torch.float32).view(1, 1, 4, 1)
        off_lr = torch.tensor([half_width, -half_width, half_width, -half_width], device=self.device, dtype=torch.float32).view(1, 1, 4, 1)
        corners_xy = feet_xy.unsqueeze(2) + foot_fwd.unsqueeze(2) * off_fb + foot_lat.unsqueeze(2) * off_lr
        corners_h = self._get_terrain_heights(
            None,
            corners_xy[..., 0].reshape(-1),
            corners_xy[..., 1].reshape(-1),
            interpolate=False
        ).view(self.num_envs, self.feet_num, 4)

        top_h = corners_h.max(dim=-1).values
        # 后期略收紧“贴台阶面”的高度容差
        support_tol = 0.015 - 0.004 * prog
        close_to_top = torch.sigmoid((corners_h - (top_h.unsqueeze(-1) - support_tol.unsqueeze(-1))) / 0.0035)
        support_ratio = close_to_top.mean(dim=-1)  # 1=明显在台阶内，0.5≈边缘，<0.5 越界

        # 边缘为 0 的内外符号：inside >0, edge ~=0, outside <0
        edge_signed = torch.clamp((support_ratio - 0.5) / 0.5, min=-1.0, max=1.0)
        center_gap = torch.clamp(top_h - ankle_h, min=0.0)
        center_on_top = torch.sigmoid((0.010 - center_gap) / 0.004)
        signed_inout = torch.clamp(edge_signed, min=0.0) * center_on_top - \
                       torch.clamp(-edge_signed, min=0.0) * (0.45 + 0.55 * torch.sigmoid((center_gap - 0.015) / 0.006))

        # 四角一致性快筛：不一致说明在边界/台阶外，没必要做重检测
        h_range = corners_h.amax(dim=-1) - corners_h.amin(dim=-1)
        flat_candidate = h_range < (0.015 - 0.004 * prog)

        # 根据脚朝向确定前后轴（x/y）
        step = float(self.cfg.terrain.horizontal_scale)
        border = float(self.cfg.terrain.border_size)

        # 前后轴只处理对应坐标：若前后轴是 x，只处理 x；若是 y，只处理 y
        use_x_axis = torch.abs(foot_fwd[:, :, 0]) >= torch.abs(foot_fwd[:, :, 1])
        axis_sign = torch.where(use_x_axis, torch.sign(foot_fwd[:, :, 0]), torch.sign(foot_fwd[:, :, 1]))
        axis_sign = torch.where(torch.abs(axis_sign) < 1e-5, torch.ones_like(axis_sign), axis_sign)
        ankle_axis = torch.where(use_x_axis, feet_xy[:, :, 0], feet_xy[:, :, 1])
        toe_xy = feet_xy + foot_fwd * front_len
        toe_axis = torch.where(use_x_axis, toe_xy[:, :, 0], toe_xy[:, :, 1])

        # 兜底边缘：仅按当前轴做 0.1m 量化（查表失效时保持信号连续）
        q = (ankle_axis + border) / step
        back_base = torch.where(
            axis_sign > 0.0,
            torch.floor(q + 1e-6) * step,
            torch.ceil(q - 1e-6) * step
        ) - border
        front_base = torch.where(
            axis_sign > 0.0,
            torch.ceil(q - 1e-6) * step,
            torch.floor(q + 1e-6) * step
        ) - border
        front_base_in = axis_sign * front_base
        ankle_in = axis_sign * ankle_axis
        front_base = torch.where(front_base_in <= ankle_in + 1e-6, front_base + axis_sign * step, front_base)
        back_dist_fallback = torch.clamp(axis_sign * (ankle_axis - back_base), min=0.0)
        toe_clear_fallback = axis_sign * (front_base - toe_axis)

        back_dist = back_dist_fallback
        toe_front_clear = toe_clear_fallback

        # 查预计算边缘索引表（x/y + 正/负前向 + back/front）
        if getattr(self, "has_stair_edge_lookup", False):
            grid_rows, grid_cols = self.height_samples.shape
            x_grid = (feet_xy[:, :, 0].to(torch.float32) + border) / step
            y_grid = (feet_xy[:, :, 1].to(torch.float32) + border) / step
            x_idx = torch.clamp(torch.floor(x_grid), 0, float(grid_rows - 1)).long()
            y_idx = torch.clamp(torch.floor(y_grid), 0, float(grid_cols - 1)).long()

            if (not hasattr(self, "rect_lateral_offsets")) or (self.rect_lateral_offsets.device != self.device):
                self.rect_lateral_offsets = torch.tensor([-1, 0, 1], device=self.device, dtype=torch.long)
            lat_offsets = self.rect_lateral_offsets
            k = int(lat_offsets.numel())

            x_idx_x = x_idx.unsqueeze(-1).expand(-1, -1, k)
            y_idx_x = torch.clamp(y_idx.unsqueeze(-1) + lat_offsets.view(1, 1, -1), 0, grid_cols - 1)
            x_idx_y = torch.clamp(x_idx.unsqueeze(-1) + lat_offsets.view(1, 1, -1), 0, grid_rows - 1)
            y_idx_y = y_idx.unsqueeze(-1).expand(-1, -1, k)

            back_idx_cand = torch.full((self.num_envs, self.feet_num, k), -1, device=self.device, dtype=torch.int32)
            front_idx_cand = torch.full((self.num_envs, self.feet_num, k), -1, device=self.device, dtype=torch.int32)

            mask_x_pos = use_x_axis & (axis_sign > 0.0)
            mask_x_neg = use_x_axis & (axis_sign < 0.0)
            mask_y_pos = (~use_x_axis) & (axis_sign > 0.0)
            mask_y_neg = (~use_x_axis) & (axis_sign < 0.0)

            if mask_x_pos.any():
                back_idx_cand[mask_x_pos] = self.stair_edge_back_x_pos[x_idx_x[mask_x_pos], y_idx_x[mask_x_pos]].to(torch.int32)
                front_idx_cand[mask_x_pos] = self.stair_edge_front_x_pos[x_idx_x[mask_x_pos], y_idx_x[mask_x_pos]].to(torch.int32)
            if mask_x_neg.any():
                back_idx_cand[mask_x_neg] = self.stair_edge_back_x_neg[x_idx_x[mask_x_neg], y_idx_x[mask_x_neg]].to(torch.int32)
                front_idx_cand[mask_x_neg] = self.stair_edge_front_x_neg[x_idx_x[mask_x_neg], y_idx_x[mask_x_neg]].to(torch.int32)
            if mask_y_pos.any():
                back_idx_cand[mask_y_pos] = self.stair_edge_back_y_pos[x_idx_y[mask_y_pos], y_idx_y[mask_y_pos]].to(torch.int32)
                front_idx_cand[mask_y_pos] = self.stair_edge_front_y_pos[x_idx_y[mask_y_pos], y_idx_y[mask_y_pos]].to(torch.int32)
            if mask_y_neg.any():
                back_idx_cand[mask_y_neg] = self.stair_edge_back_y_neg[x_idx_y[mask_y_neg], y_idx_y[mask_y_neg]].to(torch.int32)
                front_idx_cand[mask_y_neg] = self.stair_edge_front_y_neg[x_idx_y[mask_y_neg], y_idx_y[mask_y_neg]].to(torch.int32)

            inf = torch.full((self.num_envs, self.feet_num, k), 1e6, device=self.device, dtype=torch.float32)

            back_edge_axis_cand = back_idx_cand.to(torch.float32) * step - border
            back_dist_cand = axis_sign.unsqueeze(-1) * (ankle_axis.unsqueeze(-1) - back_edge_axis_cand)
            valid_back = (back_idx_cand >= 0) & (back_dist_cand >= 0.0)
            best_back_dist = torch.min(torch.where(valid_back, back_dist_cand, inf), dim=-1).values
            has_back = best_back_dist < 1e5

            front_edge_axis_cand = front_idx_cand.to(torch.float32) * step - border
            toe_clear_cand = axis_sign.unsqueeze(-1) * (front_edge_axis_cand - toe_axis.unsqueeze(-1))
            valid_front = (front_idx_cand >= 0) & (toe_clear_cand >= 0.0)
            best_toe_clear = torch.min(torch.where(valid_front, toe_clear_cand, inf), dim=-1).values
            has_front = best_toe_clear < 1e5

            use_lookup = flat_candidate
            back_dist = torch.where(use_lookup & has_back, best_back_dist, back_dist_fallback)
            toe_front_clear = torch.where(use_lookup & has_front, best_toe_clear, toe_clear_fallback)

        # 踝点到“后向台阶边缘线(x=c 或 y=c)”的垂直距离
        d = torch.clamp(back_dist, min=0.0, max=feasible_cap)
        # 在 5cm 边缘缓冲带内弱化“台阶外惩罚”，避免策略在台阶前不敢迈第一步
        edge_relax = torch.sigmoid((d - feasible_lower) / 0.01)

        # 脚尖到前向立边净距：越小越危险（防脚尖贴前方垂直面）
        toe_riser_risk = torch.clamp((0.06 - toe_front_clear) / 0.06, min=0.0, max=1.8) * flat_candidate.float()
        # 超出 13cm 可行域后直接进入负分区（与悬空同类风险），而不是仅“没有正奖励”
        deep_out_pen = torch.clamp((back_dist - feasible_cap) / 0.03, min=0.0, max=1.8)

        # 5cm=0, 9cm=1, 13cm=0；使用平滑余弦单峰，避免线性拐点不连续
        rise_arg = torch.clamp((d - feasible_lower) / max(feasible_peak - feasible_lower, 1e-6), min=0.0, max=1.0)
        fall_arg = torch.clamp((d - feasible_peak) / max(feasible_cap - feasible_peak, 1e-6), min=0.0, max=1.0)
        rise = 0.5 - 0.5 * torch.cos(math.pi * rise_arg)
        fall = 0.5 + 0.5 * torch.cos(math.pi * fall_arg)
        dist_score = torch.where(d <= feasible_peak, rise, fall)
        score_window = ((d >= feasible_lower) & (d <= feasible_cap)).float()
        dist_score = dist_score * score_window
        dist_score = torch.clamp(dist_score, min=0.0, max=1.0)
        signed_rect = torch.where(signed_inout >= 0.0, signed_inout * dist_score, signed_inout)

        # 相位/受力门控：支撑相奖励更可信；摆动末段只提前暴露负惩罚
        is_stance = (self.leg_phase < 0.55).float()
        is_pre_land = torch.clamp((self.leg_phase - 0.90) / 0.10, min=0.0, max=1.0)
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_w = torch.clamp(contact_force_z / 55.0, 0.0, 1.0)

        cmd_forward_soft = self._get_cmd_forward_soft_gate(x_thr=0.06, planar_thr=0.05)
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            _, _, _, _, front_delta, underfoot_range, _ = self._get_height_roi_features()
            near_front_conf = torch.sigmoid((front_delta - 0.008) / 0.004)
            near_under_conf = torch.sigmoid((underfoot_range - 0.032) / 0.008)
            stair_conf_eff = torch.maximum(stair_conf, torch.maximum(near_front_conf, near_under_conf))
            near_stair_soft = (stair_conf_eff > 0.16).float() * cmd_forward_soft
        else:
            stair_conf = torch.zeros(self.num_envs, device=self.device)
            stair_conf_eff = stair_conf
            near_stair_soft = torch.zeros(self.num_envs, device=self.device)

        planner_stair_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, "foothold_plan_edge_count"):
            planner_stair_mask = planner_stair_mask | (torch.max(self.foothold_plan_edge_count, dim=1).values > 0.5)
        if hasattr(self, "foothold_plan_top_out"):
            planner_stair_mask = planner_stair_mask | torch.any(self.foothold_plan_top_out, dim=1)
        stair_gate = torch.maximum(planner_stair_mask.float(), near_stair_soft)

        # 关键修复：零/弱命令下关闭 rect 正向收益，避免平台中心“为拿对齐分而慢速自旋”
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        cmd_active = (cmd_planar > 0.08).float()
        cmd_fwd_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.06)
        pos_cmd_gate = cmd_active * cmd_fwd_conf
        pos_branch = torch.clamp(signed_rect, min=0.0) * is_stance * (0.30 + 0.70 * contact_w) \
            * pos_cmd_gate.unsqueeze(1)
        neg_branch = torch.clamp(-signed_rect, min=0.0) * torch.maximum(
            is_stance * (0.45 + 0.55 * contact_w),
            0.35 * is_pre_land
        ) * (0.20 + 0.80 * edge_relax)

        # 零指令保持态：rect 全静音，避免相位/几何惩罚驱动原地摆腿。
        no_cmd_mask = self._get_no_cmd_mask()
        no_cmd_rect_scale = torch.where(no_cmd_mask, torch.zeros_like(cmd_planar), torch.ones_like(cmd_planar))

        foot_signal = pos_branch - neg_branch \
            - 0.62 * torch.clamp(signed_inout, min=0.0) * toe_riser_risk \
            - 0.60 * deep_out_pen * torch.maximum(is_stance, 0.35 * is_pre_land)

        return torch.sum(foot_signal, dim=1) * stair_gate * no_cmd_rect_scale

    def _clearance_xy_track_reward(self, foot_xy, traj_xy, phase_t):
        """
        clearance 内部使用的 XY 轨迹跟踪项：
        输入轨迹点与当前脚位，返回 XY 跟踪奖励与误差，便于单独诊断。
        """
        xy_err = torch.norm(foot_xy - traj_xy, dim=2)
        xy_sigma = 0.060
        xy_reward = torch.exp(-torch.square(xy_err) / (2.0 * xy_sigma * xy_sigma))
        xy_mid_gate = torch.clamp(torch.sin(phase_t * np.pi), min=0.0, max=1.0)
        xy_reward = xy_reward * (0.35 + 0.65 * xy_mid_gate)
        return xy_reward, xy_err

    def _clearance_z_track_reward(self, foot_h, base_traj_h, safe_traj_h, front_clear_gate):
        """
        clearance 内部使用的 Z 轨迹跟踪项：
        返回 Z 奖励主项、Z 误差、下穿惩罚、过高惩罚。
        """
        band_h = torch.clamp(safe_traj_h - base_traj_h, min=0.008, max=0.080)
        below_base_penalty = torch.clamp((base_traj_h - foot_h) / 0.018, min=0.0, max=1.6)
        in_band_ratio = torch.clamp((foot_h - base_traj_h) / (band_h + 1e-6), min=0.0, max=1.0)
        in_band_reward = in_band_ratio * (foot_h >= base_traj_h).float() * (foot_h <= safe_traj_h).float()

        safe_sigma = 0.012
        safe_peak_reward = torch.exp(-torch.square(foot_h - safe_traj_h) / (2.0 * safe_sigma * safe_sigma))
        safe_peak_reward = safe_peak_reward * (foot_h >= base_traj_h).float()
        above_safe_penalty = torch.clamp((foot_h - safe_traj_h) / 0.045, min=0.0, max=1.2)

        z_track_reward = (0.45 * in_band_reward + 0.55 * safe_peak_reward) * front_clear_gate
        z_err = torch.abs(foot_h - safe_traj_h)
        return z_track_reward, z_err, below_base_penalty, above_safe_penalty

    def _reward_stair_clearance(self):
        """
        摆动腿净空奖励（精简版）：
        1) 用锁存的起点/落点高度生成平滑目标高度带；
        2) 低于下沿惩罚，位于区间奖励；
        3) 仅保留脚尖前立边风险作为补充惩罚。
        """
        if not hasattr(self, "extras"):
            self.extras = {}
        if hasattr(self, "foothold_plan_active") and (not bool(self.foothold_plan_active.any().item())):
            self.extras["Diagnostics/clearance_xy_err_ema"] = float(self.clearance_xy_err_ema.item())
            self.extras["Diagnostics/clearance_z_err_ema"] = float(self.clearance_z_err_ema.item())
            return torch.zeros(self.num_envs, device=self.device)

        # --- 1. 提取真实的物理步态参数 ---
        gait_period = 0.8 
        swing_start = 0.55
        swing_duration_phase = 1.0 - swing_start  # 相位占比 0.45
        
        # 计算当前摆动进度 t (0.0 -> 1.0)
        # 注意：只要 phase < 0.55 (支撑相)，t 就会被严格 clamp 为 0.0
        t = torch.clamp((self.leg_phase - swing_start) / swing_duration_phase, min=0.0, max=1.0)
        
        # 屏蔽起步和落地的极点噪声区 (0.05~0.95)
        is_swinging = ((t > 0.05) & (t < 0.95)).float()
        if hasattr(self, "foothold_plan_active"):
            is_swinging = is_swinging * self.foothold_plan_active.float()

        # --- 2. 预测落足点 ---
        swing_duration_time = swing_duration_phase * gait_period
        time_remaining = (1.0 - t) * swing_duration_time 
        
        world_lin_vel = self.root_states[:, 7:9] 
        # ==========================================================
        # 👑 [核心修复 1] 向量级防近视：保持朝向不变，强制抬升速度模长
        
        # 1. 计算真实速度的模长 (Magnitude)
        vel_norm = torch.norm(world_lin_vel, dim=1, keepdim=True)
        
        # 2. 安全提取速度方向 (Direction)
        # 如果速度极小(<0.05m/s，说明撞墙卡死了)，使用机器人的朝向(Yaw)作为保底探索方向
        yaw = self.rpy[:, 2]
        fallback_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        
        # 提取方向 (自动处理除零风险)
        vel_dir = torch.where(vel_norm > 0.05, world_lin_vel / (vel_norm + 1e-6), fallback_dir)
        
        # 3. 软补速：仅在“有前进意图且接近楼梯”时补速，避免卡住时过激落点预测
        cmd_norm = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
        forward_conf = torch.sigmoid((cmd_norm - 0.10) / 0.05)
        if hasattr(self, "measured_heights") or getattr(self, "has_stair_edge_lookup", False):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False).unsqueeze(1)
        else:
            stair_conf = torch.zeros_like(cmd_norm)
        assist_w = forward_conf * stair_conf
        v_min = 0.35
        virtual_vel_norm = vel_norm + assist_w * torch.clamp(v_min - vel_norm, min=0.0)
        
        # 4. 合成最终的“虚拟世界速度” (方向不变，模长达标)
        virtual_world_vel = vel_dir * virtual_vel_norm
        
        raw_predicted_land_x = self.feet_pos[..., 0] + virtual_world_vel[:, 0].unsqueeze(1) * time_remaining
        raw_predicted_land_y = self.feet_pos[..., 1] + virtual_world_vel[:, 1].unsqueeze(1) * time_remaining
        predicted_land_xy = torch.stack([raw_predicted_land_x, raw_predicted_land_y], dim=-1)
        # 若 swing 开始已规划出落足锚点，则 clearance 全程跟踪该固定目标（不再每步漂移）
        if hasattr(self, "foothold_plan_target_xy") and hasattr(self, "foothold_plan_active"):
            predicted_land_xy = torch.where(
                self.foothold_plan_active.unsqueeze(-1),
                self.foothold_plan_target_xy,
                predicted_land_xy
            )
        predicted_land_x = predicted_land_xy[..., 0]
        predicted_land_y = predicted_land_xy[..., 1]

        # ==========================================================
        # 基于预计算边缘查表的“脚尖-立边”风险评估（替代左右采样侧边惩罚）
        # ==========================================================
        feet_quat = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        x_axis = torch.zeros(self.num_envs * self.feet_num, 3, device=self.device, dtype=torch.float32)
        x_axis[:, 0] = 1.0
        foot_fwd_3d = quat_apply(feet_quat.reshape(-1, 4), x_axis).view(self.num_envs, self.feet_num, 3)
        foot_fwd = foot_fwd_3d[:, :, :2]
        foot_fwd_norm = torch.norm(foot_fwd, dim=-1, keepdim=True)
        base_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1).unsqueeze(1)
        foot_fwd = torch.where(foot_fwd_norm > 1e-4, foot_fwd / (foot_fwd_norm + 1e-6), base_fwd)

        use_x_axis = torch.abs(foot_fwd[:, :, 0]) >= torch.abs(foot_fwd[:, :, 1])
        axis_sign = torch.where(use_x_axis, torch.sign(foot_fwd[:, :, 0]), torch.sign(foot_fwd[:, :, 1]))
        axis_sign = torch.where(torch.abs(axis_sign) < 1e-5, torch.ones_like(axis_sign), axis_sign)

        front_len = 0.12
        step = float(self.cfg.terrain.horizontal_scale)
        border = float(self.cfg.terrain.border_size)

        if (not hasattr(self, "rect_lateral_offsets")) or (self.rect_lateral_offsets.device != self.device):
            self.rect_lateral_offsets = torch.tensor([-1, 0, 1], device=self.device, dtype=torch.long)
        lat_offsets = self.rect_lateral_offsets

        def _query_front_clear(sample_xy, toe_xy):
            sample_axis = torch.where(use_x_axis, sample_xy[..., 0], sample_xy[..., 1])
            toe_axis = torch.where(use_x_axis, toe_xy[..., 0], toe_xy[..., 1])

            q = (sample_axis + border) / step
            front_base = torch.where(
                axis_sign > 0.0,
                torch.ceil(q - 1e-6) * step,
                torch.floor(q + 1e-6) * step
            ) - border
            sample_in = axis_sign * sample_axis
            front_in = axis_sign * front_base
            front_base = torch.where(front_in <= sample_in + 1e-6, front_base + axis_sign * step, front_base)
            toe_clear_fallback = axis_sign * (front_base - toe_axis)

            if not getattr(self, "has_stair_edge_lookup", False):
                return toe_clear_fallback

            grid_rows, grid_cols = self.height_samples.shape
            x_idx = torch.clamp(torch.floor((sample_xy[..., 0] + border) / step), 0, float(grid_rows - 1)).long()
            y_idx = torch.clamp(torch.floor((sample_xy[..., 1] + border) / step), 0, float(grid_cols - 1)).long()
            k = int(lat_offsets.numel())

            x_idx_x = x_idx.unsqueeze(-1).expand(-1, -1, k)
            y_idx_x = torch.clamp(y_idx.unsqueeze(-1) + lat_offsets.view(1, 1, -1), 0, grid_cols - 1)
            x_idx_y = torch.clamp(x_idx.unsqueeze(-1) + lat_offsets.view(1, 1, -1), 0, grid_rows - 1)
            y_idx_y = y_idx.unsqueeze(-1).expand(-1, -1, k)

            front_idx = torch.full((self.num_envs, self.feet_num, k), -1, device=self.device, dtype=torch.int32)
            mask_x_pos = use_x_axis & (axis_sign > 0.0)
            mask_x_neg = use_x_axis & (axis_sign < 0.0)
            mask_y_pos = (~use_x_axis) & (axis_sign > 0.0)
            mask_y_neg = (~use_x_axis) & (axis_sign < 0.0)

            if mask_x_pos.any():
                front_idx[mask_x_pos] = self.stair_edge_front_x_pos[x_idx_x[mask_x_pos], y_idx_x[mask_x_pos]]
            if mask_x_neg.any():
                front_idx[mask_x_neg] = self.stair_edge_front_x_neg[x_idx_x[mask_x_neg], y_idx_x[mask_x_neg]]
            if mask_y_pos.any():
                front_idx[mask_y_pos] = self.stair_edge_front_y_pos[x_idx_y[mask_y_pos], y_idx_y[mask_y_pos]]
            if mask_y_neg.any():
                front_idx[mask_y_neg] = self.stair_edge_front_y_neg[x_idx_y[mask_y_neg], y_idx_y[mask_y_neg]]

            edge_axis = front_idx.to(torch.float32) * step - border
            toe_clear_cand = axis_sign.unsqueeze(-1) * (edge_axis - toe_axis.unsqueeze(-1))
            valid = (front_idx >= 0) & (toe_clear_cand >= 0.0)
            inf = torch.full_like(toe_clear_cand, 1e6)
            best = torch.min(torch.where(valid, toe_clear_cand, inf), dim=-1).values
            has_edge = best < 1e5
            return torch.where(has_edge, best, toe_clear_fallback)

        current_feet_xy = self.feet_pos[..., :2]
        current_toe_xy = current_feet_xy + foot_fwd * front_len
        predicted_toe_xy = predicted_land_xy + foot_fwd * front_len

        current_front_clear = _query_front_clear(current_feet_xy, current_toe_xy)
        pred_front_clear = _query_front_clear(predicted_land_xy, predicted_toe_xy)
        front_edge_near_conf = torch.clamp((0.24 - current_front_clear) / 0.24, min=0.0, max=1.0)
        front_riser_penalty = torch.clamp((0.05 - pred_front_clear) / 0.05, min=0.0, max=1.5)
        front_clear_gate = torch.exp(-torch.square(torch.clamp(0.04 - pred_front_clear, min=0.0)) / (2.0 * 0.03 * 0.03))

        # --- 3. 实时采样原始地形高度 (带有突变风险的 Raw Data) ---
        raw_start_h = self._get_terrain_heights(
            None, self.feet_pos[..., 0].view(-1), self.feet_pos[..., 1].view(-1), interpolate=False
        ).view(self.num_envs, -1)
        raw_end_h = self._get_terrain_heights(
            None, predicted_land_x.view(-1), predicted_land_y.view(-1), interpolate=False
        ).view(self.num_envs, -1)
        step_up_raw = torch.clamp(raw_end_h - raw_start_h, min=0.0, max=0.25)
        raw_edge_obstacle_h = torch.clamp(
            step_up_raw * (0.35 + 0.65 * front_edge_near_conf) + 0.03 * front_riser_penalty,
            min=0.0,
            max=0.20
        )
        
        # ==========================================================
        # 👑 4. 【核心创新：双端高度锁存器 (Double-ended Height Latch)】
        # ==========================================================
        # 初始化 PyTorch 缓存变量 (只在第一次调用时创建)
        if not hasattr(self, 'cached_start_h'):
            self.cached_start_h = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.cached_end_h = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.cached_edge_obstacle_h = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            
        # 判断当前腿是否踩在地上 (t == 0.0)
        is_stance = (t == 0.0)
        
        # 魔法发生的地方：
        # 如果脚踩在地上 (is_stance=True)，不断用最新的地形高度刷新缓存。
        # 一旦脚起飞 (is_stance=False)，立即停止刷新，死死锁住起飞那一瞬间的高度！
        self.cached_start_h = torch.where(is_stance, raw_start_h, self.cached_start_h)
        self.cached_end_h = torch.where(is_stance, raw_end_h, self.cached_end_h)
        self.cached_edge_obstacle_h = torch.where(is_stance, raw_edge_obstacle_h, self.cached_edge_obstacle_h)
        # Planner 事件锁存：摆动刚起飞时，强制采用规划器的起点/终点/边缘风险
        # 保证 clearance 与 planner 轨迹语义一致，不再用旧落点的风险估计。
        if hasattr(self, "foothold_plan_new_event") and self.foothold_plan_new_event.any():
            plan_event = self.foothold_plan_new_event
            plan_start_h = self.foothold_plan_start_h if hasattr(self, "foothold_plan_start_h") else raw_start_h
            plan_end_h = self.foothold_plan_target_z if hasattr(self, "foothold_plan_target_z") else raw_end_h
            if hasattr(self, "foothold_plan_riser_h"):
                plan_edge_h = self.foothold_plan_riser_h + 0.02 * torch.clamp(self.foothold_plan_edge_count - 1.0, min=0.0, max=2.0)
            else:
                plan_edge_h = raw_edge_obstacle_h
            plan_edge_h = torch.clamp(plan_edge_h + 0.02 * front_riser_penalty, min=0.0, max=0.20)
            self.cached_start_h = torch.where(plan_event, plan_start_h, self.cached_start_h)
            self.cached_end_h = torch.where(plan_event, plan_end_h, self.cached_end_h)
            self.cached_edge_obstacle_h = torch.where(plan_event, plan_edge_h, self.cached_edge_obstacle_h)
        
        # 提取绝对稳定、无突变的起点和终点高度！
        current_terrain_h = self.cached_start_h
        if hasattr(self, "foothold_plan_target_z") and hasattr(self, "foothold_plan_active"):
            future_terrain_h = torch.where(self.foothold_plan_active, self.foothold_plan_target_z, self.cached_end_h)
        else:
            future_terrain_h = self.cached_end_h
        edge_obstacle_h = self.cached_edge_obstacle_h
        path_peak_h = torch.maximum(current_terrain_h, future_terrain_h) + edge_obstacle_h
        # ==========================================================
        
        # --- 5. 基线插值轨迹规划 ---
        # 现在这条线，宛如焊死在空间中的钢筋，绝对不会在半空中发生任何抖动或突变！
        linear_h = current_terrain_h * (1.0 - t) + future_terrain_h * t
        
        cmd_vel_norm = torch.clamp(torch.norm(self.commands[:, :2], dim=1), max=1.0).unsqueeze(1)
        cmd_forward = self.commands[:, 0].unsqueeze(1)
        # 无前进意图时应显著压低 clearance 目标轨迹，避免“零指令高抬腿”
        cmd_move_intent = torch.sigmoid((cmd_forward - 0.10) / 0.03) * torch.sigmoid((cmd_vel_norm - 0.12) / 0.04)
        
        # ================== 双轨迹：拔高后基线轨迹 + 安全高度轨迹 ==================
        # 1) 起终点连线 -> 跨边时整体拔高 -> 得到“基线轨迹”；
        # 2) 在基线轨迹上再叠加安全余量，得到“安全高度轨迹”；
        # 3) 打分分段：低于基线重罚，基线到安全高度给正奖，安全高度附近峰值。
        step_diff = future_terrain_h - current_terrain_h
        step_up_diff = torch.clamp(step_diff, min=0.0)
        if hasattr(self, "foothold_plan_edge_count"):
            edge_cross = (self.foothold_plan_edge_count > 0.5).float()
        else:
            edge_cross = (step_up_raw > 0.012).float()
        base_top = torch.maximum(current_terrain_h, future_terrain_h)
        edge_buffer = 0.006
        base_raise = torch.clamp(path_peak_h + edge_buffer - base_top, min=0.0, max=0.12) * edge_cross
        lifted_linear_h = linear_h + base_raise

        sin_term = torch.sin(t * np.pi)
        # 基线轨迹：拔高后的基线 + 小弧线
        # 零指令时不再强行给固定抬腿弧高，减少“静止也被相位拉着摆腿”
        arc_amp = 0.004 * cmd_move_intent + 0.06 * step_up_diff + 0.30 * base_raise
        base_traj_h = lifted_linear_h + arc_amp * sin_term
        # 安全轨迹：在基线轨迹之上叠加安全余量（跨边/近立边时更高）
        safety_extra = torch.clamp(
            0.012 + 0.010 * edge_cross + 0.010 * front_edge_near_conf + 0.015 * front_riser_penalty,
            min=0.010,
            max=0.060
        )
        safe_traj_h = base_traj_h + safety_extra

        # --- 7. 轨迹跟踪打分（Z 主导 + XY 辅助） ---
        current_foot_h = self.feet_pos[..., 2]
        if hasattr(self, "foothold_plan_start_xy"):
            xy_start_anchor = self.foothold_plan_start_xy
        else:
            xy_start_anchor = self.feet_pos[..., :2]
        xy_traj = xy_start_anchor + (predicted_land_xy - xy_start_anchor) * t.unsqueeze(-1)
        xy_track_reward, xy_err_traj = self._clearance_xy_track_reward(
            self.feet_pos[..., :2], xy_traj, t
        )
        z_track_reward, z_err_traj, below_base_penalty, above_safe_penalty = self._clearance_z_track_reward(
            current_foot_h, base_traj_h, safe_traj_h, front_clear_gate
        )

        front_riser_gate = torch.sigmoid((step_up_raw - 0.010) / 0.005)
        front_riser_penalty_term = front_riser_penalty * front_riser_gate

        cmd_planar_env = torch.norm(self.commands[:, :2], dim=1)
        cmd_forward_soft = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.04) * torch.sigmoid((cmd_planar_env - 0.06) / 0.04)
        motion_soft = torch.sigmoid((torch.norm(self.base_lin_vel[:, :2], dim=1) - 0.08) / 0.05)
        if hasattr(self, "measured_heights"):
            stair_conf_env = self._estimate_on_stairs_confidence(update_state=False)
        else:
            stair_conf_env = torch.zeros(self.num_envs, device=self.device)
        stair_task_gate = torch.clamp((stair_conf_env - 0.10) / 0.25, min=0.0, max=1.0)

        planner_stair_conf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if hasattr(self, "foothold_plan_edge_count"):
            planner_stair_conf = torch.maximum(
                planner_stair_conf,
                torch.clamp(torch.max(self.foothold_plan_edge_count, dim=1).values / 0.8, min=0.0, max=1.0),
            )
        if hasattr(self, "foothold_plan_top_out"):
            planner_stair_conf = torch.maximum(
                planner_stair_conf,
                torch.any(self.foothold_plan_top_out, dim=1).float(),
            )
        # 兜底：若 planner 不可用，使用楼梯置信门控
        if hasattr(self, "foothold_plan_edge_count") or hasattr(self, "foothold_plan_top_out"):
            stair_hard_conf = torch.maximum(
                planner_stair_conf,
                0.30 * torch.clamp((stair_task_gate - 0.18) / 0.50, min=0.0, max=1.0),
            )
        else:
            stair_hard_conf = torch.clamp((stair_task_gate - 0.18) / 0.50, min=0.0, max=1.0)

        # 运动任务软门控（前进/有运动意图时启用）
        task_motion_gate = torch.clamp(
            torch.maximum(cmd_forward_soft, 0.45 * motion_soft) * torch.sigmoid((cmd_planar_env - 0.06) / 0.04),
            min=0.0,
            max=1.0
        )
        # 楼梯专项软门控：前期保留底座梯度，避免“全 0 学不动”。
        late_prog = getattr(self, "current_late_stage_progress", 0.0)
        if torch.is_tensor(late_prog):
            late_scalar = torch.clamp(late_prog, min=0.0, max=1.0)
            if late_scalar.ndim == 0:
                late_scalar = late_scalar.repeat(self.num_envs)
            else:
                late_scalar = late_scalar.to(self.device)
        else:
            late_scalar = torch.full((self.num_envs,), float(np.clip(late_prog, 0.0, 1.0)), device=self.device)
        clearance_gate_floor = 0.20 + 0.20 * (1.0 - late_scalar)  # early~0.40, late~0.20
        stair_task_w = clearance_gate_floor + (1.0 - clearance_gate_floor) * stair_hard_conf
        task_motion_gate = task_motion_gate * stair_task_w
        # 零指令硬静音：清零 clearance 全部梯度，防止相位项驱动原地摆腿
        no_cmd_mask = self._get_no_cmd_mask()
        task_motion_gate = torch.where(no_cmd_mask, torch.zeros_like(task_motion_gate), task_motion_gate)
        reward_gate = task_motion_gate.unsqueeze(1)
        penalty_gate = task_motion_gate.unsqueeze(1)
        progress_vel, _, _, _, _, _, _ = self._get_progress_velocity_state()
        clearance_progress_conf = torch.sigmoid((progress_vel - 0.06) / 0.05).unsqueeze(1)
        penalty_soft_gate = 0.30 + 0.70 * clearance_progress_conf
        # 简化门控：只在“楼梯 + 有运动意图 + 非零指令”下启用 clearance。
        # 正向和惩罚分支共用同一门控，便于解释与调参。
        clearance_signal_z = (
            0.82 * z_track_reward * reward_gate
            - 0.95 * below_base_penalty * penalty_gate * penalty_soft_gate
            - 0.18 * above_safe_penalty * penalty_gate * penalty_soft_gate
            - 0.20 * front_riser_penalty_term * penalty_gate * penalty_soft_gate
        )
        clearance_signal_xy = 0.18 * xy_track_reward * reward_gate
        clearance_signal = clearance_signal_z + clearance_signal_xy

        # 轨迹精度诊断（XY/Z 分开看）：仅在有效摆动样本上统计并做 EMA。
        active_track_mask = (is_swinging > 0.5) & (reward_gate > 0.05)
        active_count = torch.sum(active_track_mask.float())
        if active_count > 0:
            xy_err_step = torch.sum(xy_err_traj * active_track_mask.float()) / active_count
            z_err_step = torch.sum(z_err_traj * active_track_mask.float()) / active_count
            ema_alpha = 0.06
            self.clearance_xy_err_ema = (1.0 - ema_alpha) * self.clearance_xy_err_ema + \
                ema_alpha * xy_err_step.detach().unsqueeze(0)
            self.clearance_z_err_ema = (1.0 - ema_alpha) * self.clearance_z_err_ema + \
                ema_alpha * z_err_step.detach().unsqueeze(0)
        else:
            self.clearance_xy_err_ema = 0.996 * self.clearance_xy_err_ema
            self.clearance_z_err_ema = 0.996 * self.clearance_z_err_ema
        self.extras["Diagnostics/clearance_xy_err_ema"] = float(self.clearance_xy_err_ema.item())
        self.extras["Diagnostics/clearance_z_err_ema"] = float(self.clearance_z_err_ema.item())

        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return torch.sum(clearance_signal * is_swinging, dim=1) * explore_gate

    def _reward_feet_stumble(self):
        """
        [防踢边版] 监控水平力/垂直力比例，聚焦摆动期踢台阶侧边风险。
        与 foot_flatness 分工：
        - foot_flatness: 管脚掌姿态/触地放平
        - feet_stumble: 管侧向碰撞（尤其摆动期踢边）
        """
        # 1. 提取足端力
        forces = self.contact_forces[:, self.feet_indices, :]
        force_xy = torch.norm(forces[:, :, :2], dim=2)
        force_z = torch.clamp(forces[:, :, 2], min=0.0)
        
        # 2. 侧向冲击比例：中等侧擦提前触发，重度侧擦快速拉满
        stumble_ratio = force_xy / (force_z + 4.0)
        ratio_penalty = torch.clamp((stumble_ratio - 0.55) / 0.45, min=0.0, max=1.2)
        abs_side_penalty = torch.clamp((force_xy - 10.0) / 20.0, min=0.0, max=1.2)
        penalty = torch.maximum(ratio_penalty, 0.65 * abs_side_penalty)
        
        # 3. 受力阈值过滤：摆动相更敏感，支撑相更保守
        swing_conf = torch.sigmoid((self.leg_phase - 0.54) / 0.06)
        force_thr = 4.0 + 4.0 * (1.0 - swing_conf)  # swing:4N, stance:8N
        total_force_norm = torch.norm(forces, dim=2)
        force_threshold_mask = (total_force_norm > force_thr).float()

        # 4. 相位事件门控：全相位都惩罚侧撞，摆动末段/触地窗口进一步加重
        touchdown_conf = torch.clamp((self.leg_phase - 0.78) / 0.22, min=0.0, max=1.0)
        phase_gate = torch.clamp(0.55 + 0.35 * swing_conf + 0.10 * touchdown_conf, min=0.0, max=1.2)

        # 5. 稳固支撑弱放松：只做轻微下调，避免“任何时期踢边都要管”被过度稀释
        firm_support = (force_z > 14.0) & (force_z > 2.2 * force_xy) & (self.leg_phase < 0.55)
        support_relax = torch.where(firm_support, torch.full_like(force_xy, 0.65), torch.ones_like(force_xy))

        # 6. 地形门控：台阶邻域强激活，平地尽量静音
        if hasattr(self, "measured_heights"):
            _, _, _, _, front_delta, underfoot_range, _ = self._get_height_roi_features()
            underfoot_conf = torch.sigmoid((underfoot_range - 0.035) / 0.008)
            front_conf = torch.sigmoid((front_delta - 0.012) / 0.004)
            stair_like_conf = torch.maximum(underfoot_conf, front_conf)
            underfoot_gate = 0.06 + 0.94 * stair_like_conf
        else:
            underfoot_gate = torch.ones(self.num_envs, device=self.device)

        # 侧撞属于安全约束：零命令也应惩罚，但不应被相位驱动成“抖腿对抗”
        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.05)
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        cmd_activity_conf = torch.sigmoid((cmd_planar - 0.06) / 0.04)
        forward_gate = 0.75 + 0.25 * torch.maximum(cmd_forward_conf, 0.6 * cmd_activity_conf)

        no_cmd_mask = self._get_no_cmd_mask()
        no_cmd_mask_f = no_cmd_mask.unsqueeze(1).float()
        # 零指令时减弱相位敏感度，保留基础安全惩罚
        phase_gate = phase_gate * (1.0 - 0.45 * no_cmd_mask_f)
        forward_gate = torch.where(no_cmd_mask, torch.full_like(forward_gate, 0.40), forward_gate)

        stumble_term = penalty * force_threshold_mask * phase_gate * support_relax
        return torch.sum(stumble_term, dim=1) * underfoot_gate * forward_gate

    def _reward_pelvis_height(self):
        # 1. 提取盆骨绝对高度
        base_height = self.root_states[:, 2]
        
        # 2. 获取基座正下方的真实地形高度 (这才是真正的参考面)
        base_x = self.root_states[:, 0]
        base_y = self.root_states[:, 1]
        terrain_h_under_base = self._get_terrain_heights(None, base_x, base_y).view(-1)
        
        # 3. 计算相对高度
        actual_height = base_height - terrain_h_under_base
        target_height = getattr(self, "dynamic_base_height_target", self.cfg.rewards.base_height_target)
        
        error = torch.abs(actual_height - target_height)
        
        # 4. 允许 3cm 的微小死区（比之前 5cm 更严格，抑制基座起伏/劈叉压低）
        error = torch.clamp(error - 0.03, min=0.0)
        height_reward = torch.exp(-torch.square(error) / 0.01)

        # 5. 楼梯任务门控：在上楼阶段更强约束，平地只保留低强度底座
        cmd_conf = torch.sigmoid((self.commands[:, 0] - 0.06) / 0.06)
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        no_cmd_mask = self._get_no_cmd_mask()
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            height_gate = 0.25 + 0.75 * torch.maximum(stair_conf, 0.6 * cmd_conf)
            # 零指令且非楼梯任务时，降低但不关闭高度约束，避免“脚尖上翘/后仰倒”
            stair_task_mask = stair_conf > 0.18
            relax_scale = torch.where(
                no_cmd_mask & (~stair_task_mask),
                torch.full_like(height_gate, 0.45),
                torch.ones_like(height_gate),
            )
            height_gate = height_gate * relax_scale
        else:
            height_gate = 0.25 + 0.75 * cmd_conf
            height_gate = torch.where(no_cmd_mask, 0.28 * height_gate, height_gate)
        return height_reward * height_gate

    def _reward_tracking_lin_vel(self):
        """
        线速度跟踪（带角速度一致性门控）
        目的：前进拿分必须建立在“基本服从 yaw 纠偏目标”之上，抑制平地绕圈刷分。
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 低速感知：低速命令时提高细粒度辨识，高速时保持宽容，避免“慢速碎步/不动”局部最优。
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        sigma_low = 0.08
        sigma_high = float(self.cfg.rewards.tracking_sigma)
        speed_mix = torch.clamp(cmd_speed / 0.80, min=0.0, max=1.0)
        sigma_eff = sigma_low + (sigma_high - sigma_low) * speed_mix
        lin_reward = torch.exp(-lin_vel_error / torch.clamp(sigma_eff, min=1e-4))
        # 零指令硬软结合门控：cmd<0.05 时直接不给线速度正奖
        cmd_norm = cmd_speed
        cmd_gate = torch.clamp((cmd_norm - 0.05) / 0.10, min=0.0, max=1.0)

        yaw_target = self.yaw_track_target if hasattr(self, "yaw_track_target") else self.commands[:, 2]
        yaw_err = torch.abs(yaw_target - self.base_ang_vel[:, 2])
        # 误差越大门控越低；提高底座避免“姿态一差就前进梯度几乎归零”
        yaw_align_gate = 0.55 + 0.45 * torch.sigmoid((0.35 - yaw_err) / 0.10)

        forward_mask = (self.commands[:, 0] > 0.1).float()
        gate = forward_mask * yaw_align_gate + (1.0 - forward_mask)

        # 平地抗转圈：在非楼梯区若偏航角速度过大，削弱线速度刷分收益
        progress_vel, _, v_lat, _, stair_conf, _, _ = self._get_progress_velocity_state()
        flat_conf = 1.0 - stair_conf
        spin_conf = torch.sigmoid((torch.abs(self.base_ang_vel[:, 2]) - 0.30) / 0.10)
        anti_spin_gate = 1.0 - 0.22 * flat_conf * spin_conf * forward_mask

        # 台阶邻域反怠速：临近楼梯却低推进时，线速度得分不能过高
        near_stair_conf = torch.sigmoid((stair_conf - 0.10) / 0.08)
        progress_conf = torch.sigmoid((progress_vel - 0.10) / 0.06)
        # 放松低推进抑制，避免“慢下来->前进奖励被关掉->更慢”的自锁
        anti_stall_gate = 1.0 - 0.10 * near_stair_conf * (1.0 - progress_conf) * forward_mask

        axis_soft_gain, _, _, _ = self._get_stair_axis_soft_gain(stair_conf, min_gain=0.72)
        # 零命令反漂移：不给 tracking 正分还不够，需要在零命令下对漂移给负梯度
        drift_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        drift_pen = torch.clamp((drift_speed - 0.03) / 0.12, min=0.0, max=1.2)
        fwd_drift_pen = torch.clamp((torch.abs(self.base_lin_vel[:, 0]) - 0.02) / 0.10, min=0.0, max=1.2)
        zero_cmd_pen = (1.0 - cmd_gate) * (0.65 * drift_pen + 0.35 * fwd_drift_pen)
        # 台阶横漂惩罚：抑制“斜着上/沿直角边擦行”
        lat_speed_abs = torch.abs(v_lat)
        lat_drift_pen = torch.clamp((lat_speed_abs - 0.06) / 0.20, min=0.0, max=1.2)
        axis_lat_pen = lat_drift_pen * torch.clamp((stair_conf - 0.10) / 0.30, min=0.0, max=1.0) * forward_mask * cmd_gate
        # 前进怠速惩罚：有前进命令时，若实际速度长期明显低于命令，给显式负梯度，
        # 把策略从“原地高频小碎步”推向“产生净位移”。
        fwd_cmd = torch.clamp(self.commands[:, 0], min=0.0)
        fwd_shortfall = torch.clamp((fwd_cmd - self.base_lin_vel[:, 0] - 0.05) / 0.35, min=0.0, max=1.6)
        # 仅在中高前进命令下激活怠速惩罚，避免早期起步阶段被过度压制。
        under_speed_gate = torch.sigmoid((fwd_cmd - 0.25) / 0.05)
        under_speed_pen = forward_mask * cmd_gate * under_speed_gate * fwd_shortfall
        # reset 早期冷启动保护：先让策略完成“站稳->起步”，再逐步施加怠速/横漂负项。
        if hasattr(self, "episode_length_buf"):
            startup_gate = torch.sigmoid((self.episode_length_buf.float() - 36.0) / 8.0)
        else:
            startup_gate = torch.ones(self.num_envs, device=self.device)
        axis_lat_pen = axis_lat_pen * startup_gate
        under_speed_pen = under_speed_pen * startup_gate
        # 轴向净推进保底正激励：防止“脚步很忙但净位移很低”的刷分行为。
        axis_progress_bonus = forward_mask * cmd_gate * torch.clamp((progress_vel - 0.04) / 0.22, min=0.0, max=1.0)
        return lin_reward * gate * anti_spin_gate * anti_stall_gate * axis_soft_gain * cmd_gate + \
            0.30 * axis_progress_bonus - 0.45 * zero_cmd_pen - 0.18 * axis_lat_pen - 0.18 * under_speed_pen

    def _reward_tracking_ang_vel(self):
        """
        [角速度跟踪修正版]
        目标：
        1. 小误差区保持高精度（沿用高斯形状）。
        2. 大误差区保留梯度并保持单调下降，避免“误差更大反而加分”。
        """
        # 使用 callback 产出的唯一真值目标，避免“控制目标”和“奖励目标”不一致
        tracking_target = self.yaw_track_target if hasattr(self, "yaw_track_target") else self.commands[:, 2]
        yaw_err = torch.abs(tracking_target - self.base_ang_vel[:, 2])

        # 分段连续单调：e<=e0 用高斯；e>e0 用指数尾部（更快衰减，避免大误差高分）
        e0 = 0.5
        near_reward = torch.exp(-torch.square(yaw_err) / self.cfg.rewards.tracking_sigma)
        r0 = math.exp(-(e0 * e0) / self.cfg.rewards.tracking_sigma)
        tail_tau = 0.30
        tail_reward = r0 * torch.exp(-(yaw_err - e0) / tail_tau)
        base_reward = torch.where(yaw_err <= e0, near_reward, tail_reward)

        # 零指令静止时关闭角速度正激励，避免“原地轻转也拿分”
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        yaw_cmd_mag = torch.abs(self.commands[:, 2])
        cmd_activity = torch.maximum(cmd_planar, 0.6 * yaw_cmd_mag)
        cmd_gate = torch.clamp((cmd_activity - 0.04) / 0.10, min=0.0, max=1.0)
        # 零指令额外抑制自旋：不给正向角速跟踪分还不够，需要显式惩罚 yaw 漂移。
        zero_cmd_spin_pen = (1.0 - cmd_gate) * torch.clamp(yaw_err / 0.35, min=0.0, max=2.0)
        return base_reward * cmd_gate - 0.30 * zero_cmd_spin_pen

    def _reward_stair_alignment(self):
        """
        [楼梯对齐约束]：全程生效，看到台阶线索时显著增强
        1) 全程保留低强度约束（防止“完全不学视觉”）
        2) 只要视觉中出现台阶/立面线索，即增强约束（不依赖是否已经上台阶）
        3) 在楼梯邻域继续维持较高强度，避免边界失效
        """
        if not hasattr(self, 'visual_obs_buf'):
            return torch.zeros(self.num_envs, device=self.device)

        cmd_x_gate = (self.commands[:, 0] > 0.10).float()
        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.10) / 0.05)
        lateral_task_conf = torch.sigmoid((0.14 - torch.abs(self.commands[:, 1])) / 0.05)
        # 只在“上楼任务模式”强约束；探索样本降权以减少目标冲突
        # 不再用 commands[:,2] 做门控，避免视觉纠偏变大时该项反而弱化
        forward_command_mask = cmd_x_gate * cmd_forward_conf * lateral_task_conf
        if hasattr(self, "explore_cmd_mask"):
            forward_command_mask = forward_command_mask * (~self.explore_cmd_mask).float()

        cam_cfg = self.cfg.sensor.stereo_cam
        depth_img = torch.clamp(self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width), 0.1, 2.0)

        w_start = cam_cfg.width // 3

        near_strip, far_strip = self._compute_pose_conditioned_depth_strips(depth_img)

        left_view_near = torch.mean(near_strip[:, :w_start], dim=1)
        right_view_near = torch.mean(near_strip[:, -w_start:], dim=1)
        left_view_far = torch.mean(far_strip[:, :w_start], dim=1)
        right_view_far = torch.mean(far_strip[:, -w_start:], dim=1)

        # 【视觉看歪惩罚】：左右深度不一样说明侧身走/歪着走
        vision_skew_penalty = 0.65 * torch.abs(left_view_near - right_view_near) + \
            0.35 * torch.abs(left_view_far - right_view_far)

        # 地形侧连续置信度（楼梯邻域兜底）
        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        if hasattr(self, "measured_heights"):
            _, _, _, _, front_delta, _, _ = self._get_height_roi_features()
            near_stair_conf = torch.sigmoid((front_delta - 0.008) / 0.004)
        else:
            near_stair_conf = torch.zeros_like(stair_conf)
        terrain_conf = torch.maximum(stair_conf, near_stair_conf)

        # 视觉可见性置信度：只要相机“看到前方近距立面/台阶线索”，就提升约束强度
        left_depth = torch.mean(near_strip[:, :w_start], dim=1)
        center_depth = torch.mean(near_strip[:, w_start:-w_start], dim=1)
        right_depth = torch.mean(near_strip[:, -w_start:], dim=1)
        side_depth = 0.5 * (left_depth + right_depth)
        center_near_conf = torch.sigmoid(
            (float(getattr(cam_cfg, "vision_seen_depth_thresh", 1.25)) - center_depth) /
            float(getattr(cam_cfg, "vision_seen_depth_sigma", 0.18))
        )
        center_step_conf = torch.sigmoid(
            (torch.abs(side_depth - center_depth) - float(getattr(cam_cfg, "vision_seen_edge_thresh", 0.10))) /
            float(getattr(cam_cfg, "vision_seen_edge_sigma", 0.03))
        )
        visual_seen_conf = torch.maximum(center_near_conf, center_step_conf)

        # 全程底座 + 视觉触发增强 + 楼梯邻域兜底（取最大）
        align_global_floor = float(getattr(cam_cfg, "align_global_floor", 0.12))
        gate_conf = torch.maximum(terrain_conf, visual_seen_conf)
        stair_gate = align_global_floor + (1.0 - align_global_floor) * gate_conf

        # 方向 + 幅值一致性（把“与视觉期望转向一致”显式写进奖励）
        visual_target = self.visual_target_yaw_vel if hasattr(self, "visual_target_yaw_vel") else self.commands[:, 2]
        yaw_err = torch.abs(visual_target - self.base_ang_vel[:, 2])
        mag_match = torch.exp(-torch.square(yaw_err) / 0.20)
        valid_turn = torch.abs(visual_target) > 0.03
        same_sign = torch.where(valid_turn,
                                (visual_target * self.base_ang_vel[:, 2] > 0.0).float(),
                                torch.ones_like(visual_target))
        consistency_reward = 0.5 * mag_match + 0.5 * same_sign
        consistency_penalty = 1.0 - consistency_reward

        # 梯度图轴向对齐惩罚：抑制“机身前向与楼梯轴夹角过大（斜着上）”
        stair_axis_penalty = torch.zeros(self.num_envs, device=self.device)
        lateral_drift_penalty = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "height_grad_x_map") and hasattr(self, "height_grad_y_map"):
            stair_axis, axis_reliable, corner_conf, _ = self._estimate_stair_axis_state()
            yaw = self.rpy[:, 2]
            body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
            axis_align = torch.abs(torch.sum(body_fwd * stair_axis, dim=1))
            stair_axis_gate = torch.clamp((terrain_conf - 0.10) / 0.30, min=0.0, max=1.0)
            axis_conf = torch.clamp(0.25 + 0.70 * axis_reliable + 0.45 * corner_conf, min=0.25, max=1.20)
            stair_axis_penalty = (1.0 - axis_align) * axis_conf * stair_axis_gate
            # 楼梯轴垂向漂移惩罚：角点处更严格，避免“对着直角边斜切”
            world_vel_xy = self.root_states[:, 7:9]
            stair_axis_perp = torch.stack([-stair_axis[:, 1], stair_axis[:, 0]], dim=1)
            lat_speed_abs = torch.abs(torch.sum(world_vel_xy * stair_axis_perp, dim=1))
            lateral_drift_penalty = torch.clamp((lat_speed_abs - 0.06) / 0.20, min=0.0, max=1.2) * stair_axis_gate * (0.45 + 0.55 * corner_conf)

        total_penalty = 0.30 * vision_skew_penalty + 0.20 * consistency_penalty + 0.35 * stair_axis_penalty + 0.15 * lateral_drift_penalty
        return total_penalty * forward_command_mask * stair_gate

    def _reward_stand_still(self):
        no_cmd_bool = self._get_no_cmd_mask()
        no_cmd_mask = no_cmd_bool.float()
        rewards_cfg = self.cfg.rewards

        # 1) 目标姿态误差（默认关节姿态）
        pose_err = torch.abs(self.dof_pos - self.default_dof_pos)
        joint_tol = float(getattr(rewards_cfg, "stand_still_joint_tol_rad", 0.10))
        joint_hard = float(getattr(rewards_cfg, "stand_still_joint_hard_rad", 0.30))
        joint_over = torch.clamp(pose_err - joint_tol, min=0.0)
        joint_cost = torch.mean(torch.square(joint_over), dim=1)
        joint_hard_ratio = torch.mean((pose_err > joint_hard).float(), dim=1)

        # 2) 机身姿态误差（roll/pitch 贴近期望直立）
        roll_abs = torch.abs(self.rpy[:, 0])
        pitch_abs = torch.abs(self.rpy[:, 1])
        roll_tol = float(getattr(rewards_cfg, "stand_still_roll_tol_rad", 0.10))
        pitch_tol = float(getattr(rewards_cfg, "stand_still_pitch_tol_rad", 0.10))
        roll_hard = float(getattr(rewards_cfg, "stand_still_roll_hard_rad", 0.28))
        pitch_hard = float(getattr(rewards_cfg, "stand_still_pitch_hard_rad", 0.30))
        roll_cost = torch.square(torch.clamp(roll_abs - roll_tol, min=0.0))
        pitch_cost = torch.square(torch.clamp(pitch_abs - pitch_tol, min=0.0))
        tilt_cost = roll_cost + pitch_cost
        tilt_hard = ((roll_abs > roll_hard) | (pitch_abs > pitch_hard)).float()
        # 2.1) 倾倒边界预防：接近终止阈值前就开始急剧增大代价，抑制 no_cmd 前后倾失稳
        term_roll = max(float(getattr(self.cfg.env, "terminate_roll", 0.8)), 1e-3)
        term_pitch = max(float(getattr(self.cfg.env, "terminate_pitch", 1.0)), 1e-3)
        tilt_guard_ratio = float(np.clip(getattr(rewards_cfg, "stand_still_tilt_guard_ratio", 0.72), 0.40, 0.95))
        roll_guard = torch.clamp(
            (roll_abs - tilt_guard_ratio * term_roll) / max((1.0 - tilt_guard_ratio) * term_roll, 1e-4),
            min=0.0,
            max=1.5,
        )
        pitch_guard = torch.clamp(
            (pitch_abs - tilt_guard_ratio * term_pitch) / max((1.0 - tilt_guard_ratio) * term_pitch, 1e-4),
            min=0.0,
            max=1.5,
        )
        tilt_guard_cost = torch.square(roll_guard) + torch.square(pitch_guard)

        # 3) 静止误差（站姿正确但仍然滑动/自转也要扣分）
        lin_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        yaw_rate_abs = torch.abs(self.base_ang_vel[:, 2])
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        lin_tol = float(getattr(rewards_cfg, "stand_still_lin_speed_tol", 0.05))
        yaw_tol = float(getattr(rewards_cfg, "stand_still_yaw_rate_tol", 0.10))
        ang_xy_tol = float(getattr(rewards_cfg, "stand_still_ang_xy_tol", 0.20))
        lin_hard = float(getattr(rewards_cfg, "stand_still_lin_speed_hard", 0.20))
        yaw_hard = float(getattr(rewards_cfg, "stand_still_yaw_rate_hard", 0.35))
        ang_xy_hard = float(getattr(rewards_cfg, "stand_still_ang_xy_hard", 0.65))
        lin_cost = torch.square(torch.clamp(lin_speed - lin_tol, min=0.0))
        yaw_cost = torch.square(torch.clamp(yaw_rate_abs - yaw_tol, min=0.0))
        ang_xy_cost = torch.square(torch.clamp(ang_vel_xy - ang_xy_tol, min=0.0))
        motion_cost = lin_cost + yaw_cost + 0.8 * ang_xy_cost
        motion_hard = ((lin_speed > lin_hard) | (yaw_rate_abs > yaw_hard) | (ang_vel_xy > ang_xy_hard)).float()

        # 4) 双支撑约束：0 指令下明确偏好双脚同时支撑
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.2
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.2
        single_support = (left_contact ^ right_contact).float()
        double_flight = ((~left_contact) & (~right_contact)).float()
        double_support = (left_contact & right_contact).float()
        support_cost = 0.8 * single_support + 2.5 * double_flight

        posture_cost = (
            4.2 * joint_cost
            + 3.0 * tilt_cost
            + 2.1 * tilt_guard_cost
            + 2.4 * lin_cost
            + 2.2 * yaw_cost
            + 1.8 * ang_xy_cost
            + 2.2 * support_cost
        )

        sigma = max(float(getattr(rewards_cfg, "stand_still_pose_sigma", 0.20)), 1e-4)
        base_score = torch.exp(-posture_cost / sigma)

        # 5) 超阈值惩罚：保留但弱化为“软扣分”，避免策略因强硬扣分倾向快速重置
        hard_violation = (
            joint_hard_ratio
            + (roll_abs > roll_hard).float()
            + (pitch_abs > pitch_hard).float()
            + (lin_speed > lin_hard).float()
            + (yaw_rate_abs > yaw_hard).float()
            + (ang_vel_xy > ang_xy_hard).float()
        )
        hard_penalty = torch.clamp(hard_violation / 6.0, min=0.0, max=1.0)
        hard_penalty_scale = float(getattr(rewards_cfg, "stand_still_hard_penalty_scale", 0.12))
        hard_penalty_scale = float(np.clip(hard_penalty_scale, 0.0, 0.5))
        stand_score = torch.clamp(base_score - hard_penalty_scale * hard_penalty, min=0.0, max=1.0)

        # 细分诊断（仅统计 no_cmd 子集）：仅保留核心三项，避免面板过载。
        if not hasattr(self, "extras"):
            self.extras = {}
        no_cmd_count = torch.sum(no_cmd_mask)
        no_cmd_count_val = float(no_cmd_count.item())
        if no_cmd_count_val > 0.5:
            inv_count = 1.0 / no_cmd_count_val
            self.extras["Diagnostics/stand_score_no_cmd"] = float(torch.sum(stand_score * no_cmd_mask).item() * inv_count)
            self.extras["Diagnostics/stand_motion_cost_no_cmd"] = float(torch.sum(motion_cost * no_cmd_mask).item() * inv_count)
            self.extras["Diagnostics/stand_double_support_rate_no_cmd"] = float(torch.sum(double_support * no_cmd_mask).item() * inv_count)
        else:
            self.extras["Diagnostics/stand_score_no_cmd"] = 0.0
            self.extras["Diagnostics/stand_motion_cost_no_cmd"] = 0.0
            self.extras["Diagnostics/stand_double_support_rate_no_cmd"] = 0.0
        return no_cmd_mask * stand_score

    def _compute_torques(self, actions):
        """零指令控制隔离：先恢复稳定，再收紧动作，不影响 move/stair 主任务样本。"""
        filtered_actions = actions
        if hasattr(self, "_get_no_cmd_mask"):
            no_cmd_mask = self._get_no_cmd_mask()
        else:
            no_cmd_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if hasattr(self, "phase_mode"):
            settling_mode = 1
            hold_mode = 2
            settling_mask = self.phase_mode == settling_mode
            hold_mask = self.phase_mode == hold_mode
        else:
            settling_mask = torch.zeros_like(no_cmd_mask)
            hold_mask = torch.zeros_like(no_cmd_mask)

        if no_cmd_mask.any() or settling_mask.any() or hold_mask.any():
            filtered_actions = actions.clone()
            no_cmd_scale = float(getattr(self.cfg.commands, "no_cmd_action_scale", 0.12))
            no_cmd_recover_scale = float(getattr(self.cfg.commands, "no_cmd_recover_action_scale", no_cmd_scale))
            hold_scale = float(getattr(self.cfg.commands, "no_cmd_hold_action_scale", 0.0))
            settling_scale = float(getattr(self.cfg.commands, "no_cmd_settling_action_scale", 0.35))
            no_cmd_scale = float(np.clip(no_cmd_scale, 0.0, 1.0))
            no_cmd_recover_scale = float(np.clip(no_cmd_recover_scale, no_cmd_scale, 1.0))
            hold_scale = float(np.clip(hold_scale, 0.0, 1.0))
            settling_scale = float(np.clip(settling_scale, 0.0, 1.0))
            recover_speed = max(float(getattr(self.cfg.commands, "no_cmd_recover_speed", 0.30)), 1e-3)
            recover_yaw = max(float(getattr(self.cfg.commands, "no_cmd_recover_yaw_rate", 0.55)), 1e-3)
            pose_gain = float(np.clip(getattr(self.cfg.commands, "no_cmd_pose_action_gain", 0.40), 0.0, 2.0))
            pose_blend = float(np.clip(getattr(self.cfg.commands, "no_cmd_pose_blend", 0.18), 0.0, 1.0))
            pose_blend_settling = float(np.clip(getattr(self.cfg.commands, "no_cmd_pose_blend_settling", 0.35), 0.0, 1.0))
            pose_blend_hold = float(np.clip(getattr(self.cfg.commands, "no_cmd_pose_blend_hold", 0.50), 0.0, 1.0))

            speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
            yaw_abs = torch.abs(self.base_ang_vel[:, 2])
            speed_gate = torch.clamp(speed_xy / recover_speed, min=0.0, max=1.0)
            yaw_gate = torch.clamp(yaw_abs / recover_yaw, min=0.0, max=1.0)
            roll_abs = torch.abs(self.rpy[:, 0])
            pitch_abs = torch.abs(self.rpy[:, 1])
            term_roll = max(float(getattr(self.cfg.env, "terminate_roll", 0.8)), 1e-3)
            term_pitch = max(float(getattr(self.cfg.env, "terminate_pitch", 1.0)), 1e-3)
            tilt_norm = torch.maximum(roll_abs / term_roll, pitch_abs / term_pitch)
            tilt_gate = torch.clamp((tilt_norm - 0.45) / 0.55, min=0.0, max=1.0)
            recover_gate = torch.maximum(torch.maximum(speed_gate, yaw_gate), tilt_gate)

            action_scale = max(float(getattr(self.cfg.control, "action_scale", 0.25)), 1e-4)
            pose_action = torch.clamp(
                ((self.default_dof_pos - self.dof_pos) / action_scale) * pose_gain,
                min=-1.0,
                max=1.0,
            )

            no_cmd_run_mask = no_cmd_mask & (~settling_mask) & (~hold_mask)
            if no_cmd_run_mask.any():
                run_gate = recover_gate[no_cmd_run_mask]
                run_scale = no_cmd_scale + (no_cmd_recover_scale - no_cmd_scale) * run_gate
                filtered_actions[no_cmd_run_mask] *= run_scale.unsqueeze(1)
                run_blend = pose_blend + (1.0 - pose_blend) * run_gate
                filtered_actions[no_cmd_run_mask] = (
                    (1.0 - run_blend.unsqueeze(1)) * filtered_actions[no_cmd_run_mask]
                    + run_blend.unsqueeze(1) * pose_action[no_cmd_run_mask]
                )
            if settling_mask.any():
                settling_gate = recover_gate[settling_mask]
                settling_scale_env = settling_scale + (no_cmd_recover_scale - settling_scale) * settling_gate
                filtered_actions[settling_mask] *= settling_scale_env.unsqueeze(1)
                settling_blend = pose_blend_settling + (1.0 - pose_blend_settling) * 0.5 * settling_gate
                filtered_actions[settling_mask] = (
                    (1.0 - settling_blend.unsqueeze(1)) * filtered_actions[settling_mask]
                    + settling_blend.unsqueeze(1) * pose_action[settling_mask]
                )
            if hold_mask.any():
                filtered_actions[hold_mask] *= hold_scale
                if pose_blend_hold > 0.0:
                    filtered_actions[hold_mask] = (
                        (1.0 - pose_blend_hold) * filtered_actions[hold_mask]
                        + pose_blend_hold * pose_action[hold_mask]
                    )
        return super()._compute_torques(filtered_actions)

    def _reward_contact(self):
        """ 奖励左右腿步态交替（去重版：楼梯阶段降低与 alternation 的正向重叠） """
        pos_res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        neg_res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            stance_hit = (contact & is_stance).float()
            swing_clear = ((~contact) & (~is_stance)).float()
            swing_touch = (contact & (~is_stance)).float()
            stance_miss = ((~contact) & is_stance).float()

            # 正向：支撑相正确触地 + 摆动相干净离地
            pos_res += 0.70 * stance_hit + 0.45 * swing_clear
            # 负向：拖步/失支撑
            neg_res += 0.95 * swing_touch + 0.75 * stance_miss

        # 推进门控：有命令但不推进时，contact 不应成为主要刷分来源
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        cmd_gate = torch.clamp((cmd_speed - 0.06) / 0.32, min=0.0, max=1.0)
        no_cmd_mask = self._get_no_cmd_mask().float()
        progress_vel, _, _, _, _, _, _ = self._get_progress_velocity_state()
        progress_conf = torch.sigmoid((progress_vel - 0.07) / 0.05)
        cmd_mode_gate = 0.08 + 0.92 * cmd_gate * (0.40 + 0.60 * progress_conf)

        # 零命令下禁用相位型 contact 打分，避免摆腿振荡。
        locomotion_gate = (1.0 - no_cmd_mask) * cmd_mode_gate

        # 与 stair_alternation 去重：楼梯阶段下调 contact 的“正向刷分”，负向惩罚保留
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            stair_stage = torch.clamp((stair_conf - 0.10) / 0.50, min=0.0, max=1.0)
            stair_deconflict_gate = 1.0 - 0.45 * stair_stage
        else:
            stair_deconflict_gate = torch.ones(self.num_envs, device=self.device)

        left_fz = self.contact_forces[:, self.feet_indices[0], 2]
        right_fz = self.contact_forces[:, self.feet_indices[1], 2]
        left_contact = left_fz > 1.0
        right_contact = right_fz > 1.0
        double_flight = ((~left_contact) & (~right_contact)).float()
        cmd_forward = (self.commands[:, 0] > 0.08).float()
        hop_penalty = double_flight * ((1.0 - no_cmd_mask) * (0.15 + 0.85 * cmd_forward) + no_cmd_mask * 0.25)
        single_support = (left_contact ^ right_contact).float()
        no_cmd_support_violation = no_cmd_mask * (0.25 * single_support + 2.60 * double_flight)
        # 全局约束：任何命令阶段都不鼓励双脚同时腾空（至少保留一脚支撑）
        support_guard_violation = double_flight * ((1.0 - no_cmd_mask) + 0.35 * no_cmd_mask)
        # 移动任务下“至少一脚支撑”强化：
        # 在有效命令、楼梯阶段、低推进稳定性时，对双摆违规给更强惩罚，抑制跳步失稳。
        move_support_violation = (1.0 - no_cmd_mask) * double_flight * (
            0.40
            + 0.70 * cmd_forward
            + 0.60 * stair_stage
            + 0.30 * (1.0 - progress_conf)
        )

        # 零命令下禁用 contact 正向刷分（防原地踏步/慢速自旋局部最优）
        pos_signal = pos_res * locomotion_gate * stair_deconflict_gate * (1.0 - no_cmd_mask)
        # 零命令下关闭相位型负项，改为直接约束“双足都接触”
        neg_signal = neg_res * (0.85 + 0.15 * cmd_gate) * (1.0 - no_cmd_mask)
        return pos_signal - neg_signal \
            - 0.45 * hop_penalty \
            - 1.10 * no_cmd_support_violation \
            - 0.55 * support_guard_violation \
            - 0.95 * move_support_violation
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    # def _reward_hip_pos(self):
    #     return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    def _reward_hip_pos(self):
        """
        [恢复严格版] 动态释放腰部死区，严格根据当前的视觉指令给宽容度
        """
        error = self.dof_pos - self.default_dof_pos
        
        # 对 Roll 和 Pitch 的宽容度
        error_roll = torch.clamp(torch.abs(error[:, [1, 7]]) - 0.05, min=0.0)
        error_pitch = torch.clamp(torch.abs(error[:, [2, 8]]) - 0.2, min=0.0)
        
        # 因为 commands[:, 2] 已经是视觉算出来的、极其准确的避障转弯量了
        # 指令越大，自然给的腰部宽容度越大；指令为0，死区就严格限制在 0.05！
        yaw_vel_command = torch.abs(self.commands[:, 2]) 
        # 收紧 yaw 宽容，抑制脚尖长期内扣导致的台阶侧擦
        yaw_tolerance = 0.035 + 0.12 * yaw_vel_command.unsqueeze(1)
            
        error_yaw = torch.clamp(torch.abs(error[:, [0, 6]]) - yaw_tolerance, min=0.0)
        
        # 惩罚全部使用平方，保证平滑的梯度
        penalty_roll = torch.sum(torch.square(error_roll) * 1.0, dim=1)
        penalty_pitch = torch.sum(torch.square(error_pitch) * 1.0, dim=1) 
        penalty_yaw = torch.sum(torch.square(error_yaw) * 1.25, dim=1)
        
        return penalty_roll + penalty_pitch + penalty_yaw
