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
            raw_depth = self.depth_image_buf.clone()
            noisy_depth = self._apply_depth_noise(raw_depth)   
            # 将处理后的深度图存入 extras 用于可能的记录
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
        self.obs_history_buf = torch.zeros(self.num_envs, 150, device=self.device, dtype=torch.float)
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
            h = self.measured_heights if env_ids is None else self.measured_heights[env_ids]
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
            return torch.zeros(n, dtype=torch.float, device=self.device)
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

        return conf

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

        def safe_update(name, val, per_env=False):
            if name not in self.reward_scales:
                return
            if per_env:
                self.reward_scales[name] = val * dt
            else:
                if torch.is_tensor(val):
                    val = val.item() if val.numel() == 1 else torch.mean(val).item()
                self.reward_scales[name] = float(val) * dt

        safe_update("tracking_lin_vel", 1.20 - 0.08 * late_stage_progress, per_env=True)
        safe_update("stand_still", -0.07 - 0.05 * late_stage_progress, per_env=True)
        self.dynamic_base_height_target = 0.72 + 0.03 * float(global_progress.item())
        safe_update("orientation", -0.25 - 0.45 * late_stage_progress, per_env=True)
        # safe_update("pelvis_height", 0.08 + 0.12 * late_stage_progress, per_env=True)
        safe_update("pelvis_height", 0.10 + 0.14 * late_stage_progress, per_env=True)
        safe_update("vision_stair_drive", -0.20 - 0.45 * late_stage_progress, per_env=True)
        safe_update("ang_vel_xy", -0.045 - 0.020 * late_stage_progress, per_env=True)
        ######
        safe_update("approach_stairs", 0.40 + 0.24 * (1.0 - late_stage_progress), per_env=True)
        safe_update("tracking_ang_vel", 1.10 + 0.10 * terrain_progress, per_env=True)
        safe_update("contact", 0.12 + 0.09 * late_stage_progress, per_env=True)
        safe_update("first_step_commit", 0.22 + 0.16 * (1.0 - late_stage_progress), per_env=True)
        safe_update("lin_vel_z", -0.75 + 0.18 * late_stage_progress, per_env=True)
        safe_update("action_rate", -0.020 - 0.010 * late_stage_progress, per_env=True)
        # 台阶前“脚尖贴边”专项：中后期显著增强踩边惩罚与防绊倒惩罚
        safe_update("foot_support_rect", 0.12 + 0.55 * torch.pow(late_stage_progress, 1.35), per_env=True)
        safe_update("stair_clearance", 0.70 + 0.70 * late_stage_progress, per_env=True)
        safe_update("feet_stumble", -0.40 - 0.95 * late_stage_progress, per_env=True)
        safe_update("hip_pos", -0.22 - 0.20 * late_stage_progress, per_env=True)
        safe_update("stair_alternation", 2.20 + 2.20 * late_stage_progress, per_env=True)
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
        self.current_late_stage_progress = late_stage_progress
        
        # 【核心修正】：直接塞进全局 extras，不依赖机器人的 reset，每一步都实时更新！
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["Metrics/metrics_terrain_progress"] = torch.mean(terrain_progress).item()
        return super().compute_reward()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.update_feet_state()
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
        manual_cmd_override = bool(getattr(self.cfg.env, "manual_cmd_override", False))
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
            forward_x_conf = torch.sigmoid((forward_cmd - 0.10) / 0.05)  # 0~1
            lateral_keep_conf = torch.sigmoid((0.15 - torch.abs(self.commands[:, 1])) / 0.05)
            # 不再使用 yaw_keep_conf，避免 resample 的随机 yaw 直接影响视觉纠偏权重。
            # 视觉接管由“前进任务意图 + 视觉场景置信度”共同决定，而不是由随机 yaw 决定。
            # 探索样本不做视觉 yaw 覆写：保持其随机 yaw 指令用于探索
            forward_conf = forward_x_conf * lateral_keep_conf * non_explore_conf

            cam_cfg = self.cfg.sensor.stereo_cam
            # depth_img shape: (num_envs, 20, 32) 完美匹配 height=20, width=32
            depth_img = self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width)

            # a. 【抬头看前方】：提取画面中部（第 8~12 行），用于判断前方是否空旷
            forward_strip = torch.mean(depth_img[:, 8:12, :], dim=1)
            # 平地/登顶开阔度（软判定，0~1）
            forward_mean_depth = torch.mean(forward_strip, dim=1)
            open_conf = torch.sigmoid((forward_mean_depth - 2.6) / 0.2)

            # b. 【低头看脚下】：提取画面底部 30% 区域（第 14~20 行），专注寻找脚下的致命死角
            depth_strip = torch.mean(depth_img[:, 14:20, :], dim=1)

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

            # h. 【执行指令覆盖】
            if self.cfg.commands.heading_command:
                # Heading 模式：按前进置信度渐进接管，避免指令阈值附近硬切
                visual_heading = self.commands[:, 3] + forward_conf * smoothed_yaw_vel * self.dt
                self.commands[:, 3] = torch.where(explore_mask, self.commands[:, 3], visual_heading)
                # heading 模式下，奖励跟踪“本步视觉纠偏后的角速度目标”
                self.yaw_track_target[:] = torch.where(explore_mask, self.commands[:, 2], smoothed_yaw_vel)
            else:
                # 角速度模式下，直接接管并覆盖底层随机角速度
                # 非前进状态下做指数回零，避免历史 yaw 残留导致持续转圈
                idle_decay = 0.85
                idle_yaw = self.commands[:, 2] * idle_decay
                idle_yaw = torch.where(torch.abs(idle_yaw) < 0.02,
                                       torch.zeros_like(idle_yaw),
                                       idle_yaw)
                blended_yaw = forward_conf * smoothed_yaw_vel + (1.0 - forward_conf) * idle_yaw
                visual_cmd_yaw = torch.clamp(blended_yaw, -1.0, 1.0)
                self.commands[:, 2] = torch.where(explore_mask, old_cmd_yaw, visual_cmd_yaw)
                # yaw 模式下，奖励与执行完全对齐
                self.yaw_track_target[:] = self.commands[:, 2]

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

        # print(f"前进速度：{forward_mask}")
        # print(f"[纠偏] yaw_vel: {self.visual_target_yaw_vel.mean().item():.3f}, base_ang: {self.base_ang_vel[:, 2].mean().item():.3f}, cmd: {self.commands[:, 2].mean().item():.3f}")
        # 3. 更新腿部相位 (用于奖励函数)
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
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
        组装学生 (790维) 和 老师 (980维) 的非对称观测向量 
        学生 = 150维历史本体感受 + 640维压缩视觉特征
        老师 = 学生观测 + 187维地形高度采样 + 3维物理特权参数 (摩擦、负载、真速)
        """
        # 1. 计算当前时间相位 (2 维: sin/cos)
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # 2. 构造当前帧的本体感受数据 (50 维)
        # 顺序必须固定：线速度(3), 角速度(3), 重力投影(3), 指令(3), 关节位置(12), 关节速度(12), 上一次动作(12), 相位(2)
        current_proprio = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,           # 3
            self.base_ang_vel * self.obs_scales.ang_vel,           # 3
            self.projected_gravity,                                # 3
            self.commands[:, :3] * self.commands_scale,            # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel,                # 12
            self.actions,                                          # 12
            sin_phase,                                             # 1
            cos_phase                                              # 1
        ), dim=-1)

        # 3. 更新历史堆叠缓冲区 (50 * 3 = 150 维)
        # 如果开启噪声，仅对当前帧加噪，防止历史数据噪声累积
        if self.add_noise:
            current_proprio += (2 * torch.rand_like(current_proprio) - 1) * self.noise_scale_vec[:50]
        
        # 将新数据推入，舍弃最旧的一帧
        self.obs_history_buf = torch.cat((current_proprio, self.obs_history_buf[:, :-50]), dim=-1)

        # 4. 处理并获取归一化视觉观测 (640 维)
        if hasattr(self, 'visual_obs_buf'):
            max_range = self.cfg.sensor.stereo_cam.max_range
            # 将深度图 Clamp 在量程内并做归一化处理
            visual_obs = torch.clamp(self.visual_obs_buf, 0.0, max_range) / max_range
        else:
            visual_obs = torch.zeros((self.num_envs, 640), device=self.device)

        # 5. 【组装学生观测】 (141 + 640 = 781 维)
        self.obs_buf = torch.cat((self.obs_history_buf, visual_obs), dim=-1)

        # 6. 【组装老师观测特权信息】
        # 6.1 精准地形采样 (187 维)
        h_scale = getattr(self.obs_scales, 'height_measure', 5.0)
        # 计算相对高度差 (基座高度 - 腿长 - 地形采样点高度)
        # 腿长约 0.75m (从URDF计算: hip_pitch到ankle_roll总长)
        LEG_LENGTH = 0.75
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - LEG_LENGTH - self.measured_heights, -1, 1.) * h_scale
        
        # 6.2 物理特权信息 (3 维)
        # 维度 0: 摩擦力系数 (解决 play.py 报错：使用 getattr 保底，并从 CPU 同步到 GPU)
        friction = getattr(self, 'friction_coeffs', torch.ones(self.num_envs, 1, device='cpu'))
        self.privileged_physics_params[:, 0] = friction.to(self.device).squeeze()
        
        # 维度 1: 负载增量质量 (来自拦截函数 _process_rigid_body_props 记录的数据)
        self.privileged_physics_params[:, 1] = self.payloads.to(self.device).squeeze()
        
        # 维度 2: 真实前进 X 速度 (上帝视角绝对真值)
        self.privileged_physics_params[:, 2] = self.base_lin_vel[:, 0] * self.obs_scales.lin_vel
        
        # 7. 【最终输出】 (781 + 187 + 3 = 971 维)
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
        [终极修复版 + 全向自由探索] 指令重采样与软牵引课程
        """
        manual_cmd_override = bool(getattr(self.cfg.env, "manual_cmd_override", False))
        if self.cfg.env.test and manual_cmd_override:
            # Play 手动模式：重置时不注入自动随机/牵引指令，避免抢占 WASD
            self.commands[env_ids, 0] = 0.0
            self.commands[env_ids, 1] = 0.0
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = 0.0
            else:
                self.commands[env_ids, 2] = 0.0
            if hasattr(self, "explore_cmd_mask"):
                self.explore_cmd_mask[env_ids] = False
            return

        # 1. 基础采样：调用基类生成包含前后左右、自转的彻底随机全向指令
        super()._resample_commands(env_ids)
        
        # 默认标记为“非探索样本”，仅在探索分支显式打开
        if hasattr(self, "explore_cmd_mask"):
            self.explore_cmd_mask[env_ids] = False
        if hasattr(self, "alt_prev_lead_sign"):
            self.alt_prev_lead_sign[env_ids] = 0.0
        if hasattr(self, "alt_same_lead_steps"):
            self.alt_same_lead_steps[env_ids] = 0.0

        # 2. 【概率软牵引课程】 (仅在三维地形上生效)
        if self.cfg.terrain.mesh_type == 'trimesh':

            # 重采样时先刷新这批环境的高度，避免使用上一位置的旧测高
            if self.cfg.terrain.measure_heights and len(env_ids) > 0:
                self.measured_heights[env_ids] = self._get_heights(env_ids)

            on_stairs_mask = self._estimate_on_stairs_mask(env_ids=env_ids, update_state=True)
            force_test_stairs_mode = bool(getattr(self.cfg.env, "force_test_stairs_mode", False))
            if self.cfg.env.test and force_test_stairs_mode:
                on_stairs_mask = torch.ones_like(on_stairs_mask, dtype=torch.bool)

            # 【楼梯牵引】：100% 在楼梯上的机器人被正向牵引（严禁后退和大幅度横移）
            traction_mask = on_stairs_mask

            if traction_mask.any():
                # a. 强制下达向前的指令 (0.40 ~ max m/s)，保持推进
                self.commands[env_ids[traction_mask], 0] = torch_rand_float(
                    0.40, self.command_ranges["lin_vel_x"][1],
                    (traction_mask.sum().item(), 1), device=self.device
                ).squeeze(1)

                # b. 收紧侧移 (乘以 0.1)，防止在爬楼梯时走出夸张的”螃蟹步”
                self.commands[env_ids[traction_mask], 1] *= 0.1

                # c. 处理旋转指令
                if self.cfg.commands.heading_command:
                    from isaacgym.torch_utils import quat_apply
                    forward = quat_apply(self.base_quat[env_ids[traction_mask]], self.forward_vec[env_ids[traction_mask]])
                    current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                    self.commands[env_ids[traction_mask], 3] = current_heading
                else:
                    self.commands[env_ids[traction_mask], 2] = 0.0

            # 【平地上：前向牵引 + 少量零命令 + 全向自由探索】
            flat_mask = ~on_stairs_mask
            if flat_mask.any():
                flat_env_ids = env_ids[flat_mask]
                
                # 读取分布概率
                flat_idle_prob = float(getattr(self.cfg.commands, "flat_idle_prob", 0.12))
                flat_forward_prob = float(getattr(self.cfg.commands, "flat_forward_prob", 0.78))
                flat_yaw_range = abs(float(getattr(self.cfg.commands, "flat_yaw_range", 0.1)))

                flat_rand = torch.rand(len(flat_env_ids), device=self.device)
                idle_mask_flat = flat_rand < flat_idle_prob
                active_mask_flat = ~idle_mask_flat

                # 先处理零命令样本：专门训练“无命令就停住”
                if idle_mask_flat.any():
                    idle_env_ids = flat_env_ids[idle_mask_flat]
                    self.commands[idle_env_ids, 0] = 0.0
                    self.commands[idle_env_ids, 1] = 0.0
                    if self.cfg.commands.heading_command:
                        from isaacgym.torch_utils import quat_apply
                        forward = quat_apply(self.base_quat[idle_env_ids], self.forward_vec[idle_env_ids])
                        current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                        self.commands[idle_env_ids, 3] = current_heading
                    else:
                        self.commands[idle_env_ids, 2] = 0.0

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
                            from isaacgym.torch_utils import quat_apply
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

    def _update_contact_metrics(self, hanging_mask, contact_mask):
        """ 统计全贴合率指标 """
        with torch.no_grad():
            # 强制转为 Bool 进行位运算
            contact_bool = contact_mask[:, :, 0].bool()
            hanging_any_bool = hanging_mask.any(dim=-1).bool()
            
            # 全贴合判定：脚在地上 且 四个角都没悬空
            is_full_contact = contact_bool & (~hanging_any_bool)
            
            total_contacts = torch.sum(contact_bool).float()
            if total_contacts > 0:
                self.extras["metrics/full_contact_rate"] = torch.sum(is_full_contact).float() / total_contacts
    
    def reset_idx(self, env_ids):
        """ 重置环境时的钩子函数 """
        # 如果是在测试模式（play.py），且日志积累到一定量，则保存
        # 这里的 self.cfg.env.test 取决于你运行 play.py 时是否传入了相应参数
        # if len(self.foothold_log) > 100000: 
        #     self.save_foothold_data()
        self.obs_history_buf[env_ids] = 0
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

    def _reward_stair_traction_pull(self):
        """
        [台阶梯度牵引奖励 - 论文思想复现]
        计算机器人前方地形的“梯度（上升最快的方向）”，
        奖励机器人的朝向（Yaw）向这个牵引方向靠拢。
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)
            
        # 1. 重塑 17x11 的测高网格 (机器人在 X=8, Y=5 的位置)
        heights_grid = self.measured_heights.view(self.num_envs, 17, 11)
        
        # 2. 探测前方 0.3m 处的左、中、右高度
        # X 索引 11 大约是前方一步的距离
        front_left = heights_grid[:, 11, 8]   # 前方偏左
        front_center = heights_grid[:, 11, 5] # 正前方
        front_right = heights_grid[:, 11, 2]  # 前方偏右
        
        # 3. 寻找“最高点”（下一级台阶的牵引目标）
        # 如果左边最高，说明台阶在左前方，牵引力向左；右边同理
        # 用左右高度差来近似横向的梯度 (Gradient Y)
        grad_y = front_left - front_right
        
        # 4. 如果前方没有台阶（平地），牵引力为 0
        is_flat = torch.abs(front_center - heights_grid[:, 8, 5]) < 0.05
        
        # 5. 计算朝向纠正得分
        # 如果台阶在左边 (grad_y > 0)，我们需要机器人产生一个向左的角速度 (base_ang_vel[:, 2] > 0)
        # 乘积为正，说明机器人正在顺着牵引力转弯！给大奖！
        traction_reward = grad_y * self.base_ang_vel[:, 2]
        
        # 如果在平地上，不给这个牵引奖励
        traction_reward = traction_reward * (~is_flat).float()
        
        return traction_reward

    def _reward_approach_stairs(self):
        """
        [前置引导奖励]
        基于“当前课程等级对应的目标台阶高度”做前置引导。
        当前方高度差越接近目标台阶高度，且机器人有前进意图并实际前进时，奖励越大。
        """
        if not hasattr(self, 'measured_heights'):
            return torch.zeros(self.num_envs, device=self.device)

        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        heights = self.measured_heights.view(self.num_envs, nx, ny)
        cx, cy = nx // 2, ny // 2

        center = heights[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))
        front_near = heights[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))
        front_far = heights[:, max(0, cx + 5):min(nx, cx + 9), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))
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
        near_stair = torch.sigmoid((delta_h - 0.006) / 0.004)

        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        cmd_x_gate = (self.commands[:, 0] > 0.08).float()
        forward_cmd_conf = torch.sigmoid((self.commands[:, 0] - 0.05) / 0.06)
        lateral_task_conf = torch.sigmoid((0.14 - torch.abs(self.commands[:, 1])) / 0.05)
        # 注意：这里不再使用 commands[:,2] 做任务门控，避免其被视觉覆写后
        # 在“需要较大转向纠偏”场景下反向削弱前置牵引奖励。
        task_mode_conf = forward_cmd_conf * lateral_task_conf
        progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.03) / 0.08)
        # 台阶前卡住时保留推进牵引，不让奖励在低速时近似掉光
        vel_gate = 0.12 + 0.88 * progress_conf
        # 保留“非楼梯优先”语义，同时在已上楼梯时仅保留很小尾巴，避免“台阶前刷分”主导
        # stair_conf=0 -> gate=1.0；stair_conf=1 -> gate=0.10
        gate = torch.clamp(0.10 + 0.90 * (1.0 - stair_conf), min=0.10, max=1.0)

        # 视觉可见台阶时增强牵引：解决“平台中心已看见台阶但牵引仍偏弱”
        if hasattr(self, "visual_obs_buf") and hasattr(self.cfg, "sensor") and self.cfg.sensor.stereo_cam.enable:
            cam_cfg = self.cfg.sensor.stereo_cam
            depth_img = self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width)
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
        seen_boost = 0.10 + 0.90 * visual_seen_conf

        # 探索样本降权（不是清零）：避免与“上楼任务奖励”冲突，同时保留少量泛化梯度
        task_gate = cmd_x_gate * (0.05 + 0.95 * task_mode_conf)
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return height_match * near_stair * task_gate * vel_gate * gate * seen_boost * explore_gate

    def _reward_first_step_commit(self):
        """
        台阶前“第一步承诺奖励”：
        在临近楼梯且有前进意图时，鼓励出现“前足前探 + 双脚高度差接近目标台阶高”的动作，
        避免停在台阶前原地踏步。
        """
        if not hasattr(self, 'measured_heights'):
            return torch.zeros(self.num_envs, device=self.device)

        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        heights = self.measured_heights.view(self.num_envs, nx, ny)
        cx, cy = nx // 2, ny // 2
        center = heights[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))
        front = heights[:, max(0, cx + 2):min(nx, cx + 7), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))
        delta_h = torch.clamp(front - center, min=0.0, max=0.30)

        near_stair_conf = torch.sigmoid((delta_h - 0.010) / 0.006)
        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        pre_stair_gate = near_stair_conf * (1.0 - torch.clamp((stair_conf - 0.45) / 0.18, min=0.0, max=1.0))

        cmd_x_gate = (self.commands[:, 0] > 0.10).float()
        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.06)
        lateral_task_conf = torch.sigmoid((0.14 - torch.abs(self.commands[:, 1])) / 0.05)
        # 不依赖 commands[:,2]（该通道会被视觉回调覆写），避免楼梯前转向阶段门控误降
        cmd_conf = cmd_x_gate * cmd_forward_conf * lateral_task_conf
        progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.03) / 0.07)
        progress_gate = 0.10 + 0.90 * progress_conf

        rows = float(max(self.cfg.terrain.num_rows, 1))
        difficulty = self.terrain_levels.float() / rows
        target_step_height = 0.05 + 0.1 * difficulty
        z_diff = torch.abs(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2])
        z_match = torch.exp(-torch.square(z_diff - target_step_height) / (2.0 * 0.05 * 0.05))

        yaw = self.rpy[:, 2]
        root_x = self.root_states[:, 0].unsqueeze(1)
        root_y = self.root_states[:, 1].unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        foot_dx_w = self.feet_pos[:, :, 0] - root_x
        foot_dy_w = self.feet_pos[:, :, 1] - root_y
        foot_x_body = cos_yaw * foot_dx_w + sin_yaw * foot_dy_w
        step_reach = torch.sigmoid((torch.max(foot_x_body, dim=1).values - 0.22) / 0.04)

        swing_motion = torch.clamp(torch.max(torch.abs(self.feet_vel[:, :, 2]), dim=1).values / 0.20, min=0.0, max=1.0)
        motion_gate = 0.05 + 0.95 * swing_motion

        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return pre_stair_gate * cmd_conf * progress_gate * motion_gate * (0.6 * z_match + 0.4 * step_reach) * explore_gate
    
    def _reward_stair_alternation(self):
        """
        [空间交替步态：强制台阶跨步版 - 修复平地刷分BUG]
        核心思想：在楼梯区域，强制要求双脚在踩实时，高度差必须等于系统生成的标准台阶高度！
        彻底杜绝在同一级台阶上“并拢双脚/僵尸跳”的偷懒行为。
        在平地上，此奖励被完全静音，防止机器人靠贴地滑行刷分。
        """
        # 1. 提取系统生成的绝对台阶高度 (上帝视角)
        # 对齐 terrain.py: difficulty = i / num_rows（i 即 terrain_levels）
        max_level = self.cfg.terrain.num_rows
        difficulty = self.terrain_levels.float() / max_level
        target_step_height = 0.05 + 0.1 * difficulty
        
        # 2. 获取双脚的实际 Z 轴高度及高差
        left_foot_z = self.feet_pos[:, 0, 2]
        right_foot_z = self.feet_pos[:, 1, 2]
        signed_z_diff = left_foot_z - right_foot_z
        z_diff = torch.abs(left_foot_z - right_foot_z)
        
        # 3. 【核心环境识别】：它现在到底在“爬楼”还是在“平台”上？
        if not hasattr(self, 'measured_heights'):
            return torch.zeros(self.num_envs, device=self.device)
            
        # 严格楼梯激活：仅依据脚下局部起伏，不看远前方，避免“未上楼先扣分”
        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        g = self.measured_heights.view(self.num_envs, nx, ny)
        cx, cy = nx // 2, ny // 2
        underfoot = g[:, max(0, cx - 2):min(nx, cx + 3), max(0, cy - 2):min(ny, cy + 3)]
        underfoot_range = underfoot.amax(dim=(1, 2)) - underfoot.amin(dim=(1, 2))
        # 保留楼梯专用语义，但从 bool 改为连续置信度门控，避免边界硬切
        stair_conf_local = torch.sigmoid((underfoot_range - 0.04) / 0.006)
        # 奖励分支早一点拉起，惩罚分支晚一点拉起，避免“刚到楼梯边就被重罚”
        bonus_stair_gate = torch.clamp((stair_conf_local - 0.08) / 0.42, min=0.0, max=1.0)
        penalty_stair_gate = torch.clamp((stair_conf_local - 0.22) / 0.50, min=0.0, max=1.0)
        
        # 4. 计算高度差误差
        diff_error = torch.abs(z_diff - target_step_height)
        
        # 5. 支撑门控（保持 0.55 语义对齐）
        # 惩罚门：严格（硬双支撑 + 硬接触）
        # 奖励门：软化（软双支撑 + 软接触），缓解信号稀疏
        phase_left = self.leg_phase[:, 0]
        phase_right = self.leg_phase[:, 1]
        left_fz = self.contact_forces[:, self.feet_indices[0], 2]
        right_fz = self.contact_forces[:, self.feet_indices[1], 2]
        min_fz = torch.minimum(left_fz, right_fz)

        # 与 _reward_contact 相位语义保持一致：phase < 0.55 视为支撑相
        hard_double_stance = (phase_left < 0.55) & (phase_right < 0.55)
        hard_contact = min_fz > 1.0

        stance_left_conf = torch.sigmoid((0.55 - phase_left) / 0.04)
        stance_right_conf = torch.sigmoid((0.55 - phase_right) / 0.04)
        soft_double_stance = torch.minimum(stance_left_conf, stance_right_conf)
        support_gate = 0.30 + 0.70 * torch.sigmoid((min_fz - 4.0) / 3.0)

        # 5.1 交替“换脚角色”约束：
        # 不仅要求 |z_diff| 接近台阶高，还要求左右脚领先角色能随相位切换。
        # 放宽角色切换识别窗口：让“换脚”约束覆盖更完整摆动段，减少单脚长期领步的 step-to 解
        left_swing_conf = torch.sigmoid((phase_left - 0.52) / 0.07)
        right_swing_conf = torch.sigmoid((phase_right - 0.52) / 0.07)
        left_lead_mode = left_swing_conf * stance_right_conf
        right_lead_mode = right_swing_conf * stance_left_conf
        role_gate = torch.clamp(torch.maximum(left_lead_mode, right_lead_mode), min=0.0, max=1.0)
        lead_conf = torch.maximum(left_lead_mode, right_lead_mode)
        lead_sign = torch.where(
            left_lead_mode > right_lead_mode + 0.04,
            torch.ones_like(role_gate),
            torch.where(
                right_lead_mode > left_lead_mode + 0.04,
                -torch.ones_like(role_gate),
                torch.zeros_like(role_gate)
            )
        )
        expected_signed_diff = (left_lead_mode - right_lead_mode) * target_step_height
        sign_error = torch.abs(signed_z_diff - expected_signed_diff)
        sign_sigma = 0.07
        sign_match_reward = torch.exp(-torch.square(sign_error) / (2.0 * sign_sigma * sign_sigma))
        
        # 6. 推进门控：交替不仅要“形似”，还要伴随实际前向推进
        cmd_x_gate = (self.commands[:, 0] > 0.08).float()
        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.06)
        progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.06) / 0.05)
        progress_gate = 0.15 + 0.85 * progress_conf
        common_forward = cmd_x_gate * torch.maximum(cmd_forward_conf, 0.6 * progress_conf)

        # 6.0 领先脚角色“切换记忆”：
        # 在楼梯前进任务下，若同一只脚持续领步过久则增惩，抑制 step-to 局部最优。
        if not hasattr(self, "alt_prev_lead_sign"):
            self.alt_prev_lead_sign = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if not hasattr(self, "alt_same_lead_steps"):
            self.alt_same_lead_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        role_update_mask = (lead_conf > 0.42) & (bonus_stair_gate > 0.15) & (common_forward > 0.45)
        prev_sign = self.alt_prev_lead_sign
        same_lead = role_update_mask & (torch.abs(prev_sign) > 0.5) & (lead_sign == prev_sign) & (lead_sign != 0.0)
        switched_lead = role_update_mask & (torch.abs(prev_sign) > 0.5) & (lead_sign != 0.0) & (lead_sign != prev_sign)

        next_hold = torch.where(
            same_lead,
            self.alt_same_lead_steps + 1.0,
            torch.where(switched_lead, torch.zeros_like(self.alt_same_lead_steps), self.alt_same_lead_steps)
        )
        inactive_reset = (bonus_stair_gate < 0.08) | (common_forward < 0.30)
        self.alt_same_lead_steps = torch.where(inactive_reset, torch.zeros_like(next_hold), next_hold)

        next_prev_sign = torch.where(role_update_mask & (lead_sign != 0.0), lead_sign, prev_sign)
        self.alt_prev_lead_sign = torch.where(inactive_reset, torch.zeros_like(next_prev_sign), next_prev_sign)
        same_lead_penalty = torch.clamp((self.alt_same_lead_steps - 10.0) / 14.0, min=0.0, max=1.2) * role_update_mask.float()

        # 6.1 步宽/不交叉约束：抑制脚尖内扣、双脚过窄导致的蹭台阶侧面
        yaw = self.rpy[:, 2]
        root_x = self.root_states[:, 0].unsqueeze(1)
        root_y = self.root_states[:, 1].unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        foot_dx_w = self.feet_pos[:, :, 0] - root_x
        foot_dy_w = self.feet_pos[:, :, 1] - root_y
        foot_y_body = -sin_yaw * foot_dx_w + cos_yaw * foot_dy_w
        # 左脚应在 +y，右脚应在 -y；step_width 越小越容易内扣/互相干涉
        step_width = foot_y_body[:, 0] - foot_y_body[:, 1]
        narrow_penalty = torch.clamp((0.14 - step_width) / 0.06, min=0.0, max=1.2)
        cross_penalty = torch.clamp((-step_width) / 0.05, min=0.0, max=1.2)
        lateral_penalty = (0.7 * narrow_penalty + 0.3 * cross_penalty) * common_forward * bonus_stair_gate
        double_flight = ((left_fz < 1.0) & (right_fz < 1.0)).float()
        hop_penalty = double_flight * common_forward * bonus_stair_gate * (0.25 + 0.75 * (1.0 - progress_conf))
        
        # 7. 误差惩罚 + 匹配奖励（返回“奖励-惩罚”，可正可负）
        # 为避免课程后期“负向尾巴过重”压制上楼动机：
        # - 惩罚项加小死区并限幅
        # - 最终信号做对称限幅
        norm_error = diff_error / (target_step_height + 1e-4)
        penalty_term = torch.clamp((norm_error - 0.10) / 1.10, min=0.0, max=1.2)

        sigma = 0.24
        match_reward = torch.exp(-torch.square(norm_error) / (2.0 * sigma * sigma))
        # 以 signed 约束为主：即使在 role_gate 较低阶段也保留一部分“左右角色”信息，
        # 避免长期退化为单脚领步、另一脚跟随的 step-to 策略。
        role_blend = 0.08 + 0.92 * role_gate
        alternation_reward = (1.0 - role_blend) * match_reward + role_blend * sign_match_reward
        bonus_weight = 1.55
        penalty_gate = common_forward * penalty_stair_gate * hard_double_stance.float() * hard_contact.float()
        bonus_gate = common_forward * bonus_stair_gate * soft_double_stance * support_gate * progress_gate

        penalty_signal = penalty_term * penalty_gate
        bonus_signal = bonus_weight * alternation_reward * bonus_gate
        stall_penalty = (1.0 - progress_conf) * common_forward * bonus_stair_gate * support_gate * 0.40
        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return torch.clamp(
            bonus_signal - penalty_signal - stall_penalty - 0.35 * lateral_penalty - 0.35 * same_lead_penalty - 0.30 * hop_penalty,
            min=-1.1,
            max=1.1
        ) * explore_gate

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
        pre_land_swing = torch.clamp((self.leg_phase - 0.85) / 0.15, min=0.0, max=1.0) * is_pure_swing
        swing_penalty = torch.clamp(horizontal_deviation - 0.25, min=0.0) * (0.4 * pre_land_swing)
        
        # 4. 软门控：全程保留脚掌姿态约束，楼梯段适度增强
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            flatness_gate = 0.7 + 0.4 * stair_conf  # [0.7, 1.1]
        else:
            flatness_gate = torch.ones(self.num_envs, device=self.device)

        return torch.sum(stance_penalty + swing_penalty, dim=1) * flatness_gate

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

        # 脚本体朝向（投影到水平面），比“机体前向”更能兼容脚斜着落地
        feet_quat = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        x_axis = torch.zeros(self.num_envs * self.feet_num, 3, device=self.device, dtype=torch.float32)
        x_axis[:, 0] = 1.0
        foot_fwd_3d = quat_apply(feet_quat.reshape(-1, 4), x_axis).view(self.num_envs, self.feet_num, 3)
        foot_fwd = foot_fwd_3d[:, :, :2]
        foot_fwd_norm = torch.norm(foot_fwd, dim=-1, keepdim=True)
        yaw = self.rpy[:, 2]
        base_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1).unsqueeze(1)
        foot_fwd = torch.where(foot_fwd_norm > 1e-4, foot_fwd / (foot_fwd_norm + 1e-6), base_fwd)
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
        toe_riser_risk = torch.clamp((0.05 - toe_front_clear) / 0.05, min=0.0, max=1.5) * flat_candidate.float()

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

        pos_branch = torch.clamp(signed_rect, min=0.0) * is_stance * (0.30 + 0.70 * contact_w)
        neg_branch = torch.clamp(-signed_rect, min=0.0) * torch.maximum(
            is_stance * (0.45 + 0.55 * contact_w),
            0.25 * is_pre_land
        ) * (0.25 + 0.75 * edge_relax)
        foot_signal = pos_branch - neg_branch - 0.35 * torch.clamp(signed_inout, min=0.0) * toe_riser_risk

        # 楼梯门控：平地基本静音，台阶邻域逐步拉起
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            stair_gate = 0.05 + 0.95 * torch.clamp((stair_conf - 0.10) / 0.45, min=0.0, max=1.0)
        else:
            stair_gate = torch.ones(self.num_envs, device=self.device)

        return torch.sum(foot_signal, dim=1) * stair_gate

    def _reward_stair_clearance(self):
        """
        [终极真理版 - 状态机锁存轨迹规划] 
        核心逻辑：
        1. 严格基于底层步态周期计算剩余飞行时间。
        2. 采用用户提出的“状态机缓存”思想，在脚起跳的瞬间，用 PyTorch 变量死死锁住“起跳高度”和“预测落点高度”。
        3. 彻底根除任何因脚部越界或速度波动导致的目标轨迹空中突变！
        """
        # --- 1. 提取真实的物理步态参数 ---
        gait_period = 0.8 
        swing_start = 0.55
        swing_duration_phase = 1.0 - swing_start  # 相位占比 0.45
        
        # 计算当前摆动进度 t (0.0 -> 1.0)
        # 注意：只要 phase < 0.55 (支撑相)，t 就会被严格 clamp 为 0.0
        t = torch.clamp((self.leg_phase - swing_start) / swing_duration_phase, min=0.0, max=1.0)
        
        # 屏蔽起步和落地的极点噪声区 (0.05~0.95)
        is_swinging = ((t > 0.05) & (t < 0.95)).float()

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
        
        predicted_land_x = self.feet_pos[..., 0] + virtual_world_vel[:, 0].unsqueeze(1) * time_remaining
        predicted_land_y = self.feet_pos[..., 1] + virtual_world_vel[:, 1].unsqueeze(1) * time_remaining
        root_x = self.root_states[:, 0].unsqueeze(1)
        root_y = self.root_states[:, 1].unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)

        pred_dx_w = predicted_land_x - root_x
        pred_dy_w = predicted_land_y - root_y
        pred_dx_b = cos_yaw * pred_dx_w + sin_yaw * pred_dy_w
        pred_dy_b = -sin_yaw * pred_dx_w + cos_yaw * pred_dy_w

        curr_dx_w = self.feet_pos[..., 0] - root_x
        curr_dy_w = self.feet_pos[..., 1] - root_y
        curr_dx_b = cos_yaw * curr_dx_w + sin_yaw * curr_dy_w
        curr_dy_b = -sin_yaw * curr_dx_w + cos_yaw * curr_dy_w
        landing_xy_err = torch.sqrt(torch.square(curr_dx_b - pred_dx_b) + torch.square(curr_dy_b - pred_dy_b) + 1e-6)
        landing_xy_match = torch.exp(-torch.square(landing_xy_err) / (2.0 * 0.16 * 0.16))
        # 预测落足点的步宽走廊：抑制“脚尖过于靠内”的落足倾向
        desired_lateral = torch.tensor([0.125, -0.125], device=self.device, dtype=torch.float32).unsqueeze(0)
        pred_lateral_err = torch.abs(pred_dy_b - desired_lateral)
        lateral_lane_match = torch.exp(-torch.square(pred_lateral_err) / (2.0 * 0.045 * 0.045))

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
        predicted_land_xy = torch.stack([predicted_land_x, predicted_land_y], dim=-1)
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
        
        # 提取绝对稳定、无突变的起点和终点高度！
        current_terrain_h = self.cached_start_h
        future_terrain_h = self.cached_end_h
        edge_obstacle_h = self.cached_edge_obstacle_h
        path_peak_h = torch.maximum(current_terrain_h, future_terrain_h) + edge_obstacle_h
        # ==========================================================
        
        # --- 5. 基线插值轨迹规划 ---
        # 现在这条线，宛如焊死在空间中的钢筋，绝对不会在半空中发生任何抖动或突变！
        linear_h = current_terrain_h * (1.0 - t) + future_terrain_h * t
        
        cmd_vel_norm = torch.clamp(torch.norm(self.commands[:, :2], dim=1), max=1.0).unsqueeze(1)
        
        # ================== 地形感知动态解耦 ==================
        step_diff = future_terrain_h - current_terrain_h
        
        # 仅在上坡/上台阶时提取正向高度差，平地和下坡为 0
        step_up_diff = torch.clamp(step_diff, min=0.0) 
        
        # 双目标抬腿带：低目标用于欠高惩罚，高目标用于奖励封顶
        # 若路径中间有更高障碍（台阶立面/棱边），额外提高轨迹顶点
        mid_obstacle_h = torch.clamp(edge_obstacle_h, min=0.0, max=0.20)
        # 轻量引导：保留中段障碍抬高，但降低“超额抬腿”驱动力，避免跳步/探路脚固化
        base_amplitude = 0.05 * cmd_vel_norm + 0.5 * step_up_diff + 0.45 * mid_obstacle_h
        low_height = 0.01 + base_amplitude
        high_height = 0.03 + base_amplitude

        sin_term = torch.sin(t * np.pi)
        sine_lift_low = low_height * sin_term
        sine_lift_high = high_height * sin_term

        target_h_low = linear_h + sine_lift_low
        target_h_high = linear_h + sine_lift_high
        # 路径安全余量：中段轨迹至少高于路径峰值一小段，避免脚尖扫到台阶侧面
        safety_margin = 0.02
        path_needed = torch.clamp(path_peak_h + safety_margin - linear_h, min=0.0, max=0.18)
        path_lift = path_needed * torch.pow(sin_term, 1.2)
        target_h_low = target_h_low + 0.35 * path_lift
        target_h_high = target_h_high + 0.65 * path_lift
        # ====================================================
        
        # --- 7. 分段信号：低于 low 罚分，low~high 给正奖励，>=high 封顶 1.0 ---
        current_foot_h = self.feet_pos[..., 2]
        below_low_err = torch.clamp(target_h_low - current_foot_h, min=0.0)
        low_penalty = torch.clamp(below_low_err / 0.02, min=0.0, max=1.0)  # [0, 1]

        in_band_reward = torch.clamp(
            (current_foot_h - target_h_low) / (target_h_high - target_h_low + 1e-6),
            min=0.0,
            max=1.0
        )
        # 只抬高不往落足预测点推进/不在步宽走廊/贴近侧边立面时，奖励都要下降
        landing_gate = 0.05 + 0.45 * landing_xy_match + 0.50 * lateral_lane_match
        in_band_reward = in_band_reward * landing_gate * front_clear_gate

        # 奖惩分支分开门控：
        # - 奖励分支：平地严格静音，楼梯边界平滑拉起
        # - 惩罚分支：比奖励更早激活，提前约束“贴地踢阶”风险
        if hasattr(self, "measured_heights") or getattr(self, "has_stair_edge_lookup", False):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            cmd_x_gate = (self.commands[:, 0] > 0.08).float()
            cmd_norm = torch.norm(self.commands[:, :2], dim=1)
            cmd_conf = cmd_x_gate * torch.sigmoid((cmd_norm - 0.10) / 0.05)
            progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.05) / 0.05)
            progress_gate = 0.20 + 0.80 * progress_conf
            # 奖励更早拉起、惩罚稍后拉起：降低台阶前“先罚后奖”卡死概率
            reward_stair_gate = torch.clamp((stair_conf - 0.03) / 0.36, min=0.0, max=1.0)
            penalty_stair_gate = torch.clamp((stair_conf - 0.10) / 0.40, min=0.0, max=1.0)
            reward_gate = reward_stair_gate * cmd_conf * progress_gate
            penalty_gate = penalty_stair_gate * cmd_conf * (0.40 + 0.60 * progress_conf)
            front_penalty_gate = torch.clamp((stair_conf - 0.08) / 0.36, min=0.0, max=1.0) * cmd_conf * (0.55 + 0.45 * progress_conf)
            stall_penalty = (1.0 - progress_conf).unsqueeze(1) * in_band_reward * reward_stair_gate.unsqueeze(1) * cmd_conf.unsqueeze(1) * 0.40
        else:
            reward_gate = torch.ones(self.num_envs, device=self.device)
            penalty_gate = torch.ones(self.num_envs, device=self.device)
            front_penalty_gate = torch.ones(self.num_envs, device=self.device)
            stall_penalty = torch.zeros_like(in_band_reward)

        clearance_reward = in_band_reward * reward_gate.unsqueeze(1)
        clearance_penalty = low_penalty * penalty_gate.unsqueeze(1)
        front_riser_penalty_term = front_riser_penalty * front_penalty_gate.unsqueeze(1)
        # 与 foot_support_rect 的脚尖立边惩罚去冗余：这里仅保留较弱的摆动期补充约束
        clearance_signal = clearance_reward - clearance_penalty - stall_penalty - 0.18 * front_riser_penalty_term

        explore_gate = self._get_explore_task_gate(default_weight=0.30)
        return torch.sum(clearance_signal * is_swinging, dim=1) * explore_gate

    def _reward_feet_stumble(self):
        """
        [力矢量分析版] 监控水平力/垂直力比例，精准识别“踢台阶”动作。
        """
        # 1. 提取足端力 (假设 self.feet_indices 已经包含所有脚的索引)
        forces = self.contact_forces[:, self.feet_indices, :]
        force_xy = torch.norm(forces[:, :, :2], dim=2) 
        force_z = torch.abs(forces[:, :, 2])
        
        # 2. 计算绊倒比例：降低归一化分母，提升对中等侧擦的敏感度
        stumble_ratio = force_xy / (force_z + 6.0)
        
        # 3. 构造惩罚映射：对轻中度侧擦提前触发，对重度侧擦快速拉满
        # 使用 tanh 确保单步惩罚有上限值 1.0
        penalty = torch.tanh(torch.clip(stumble_ratio - 0.95, min=0.0))
        
        # 4. 物理过滤：只有总受力足够大时才计入惩罚，减少微小接触噪声影响
        total_force_norm = torch.norm(forces, dim=2)
        force_threshold_mask = (total_force_norm > 6.0).float()

        # 摆动相“踢侧面”风险更高：对摆动腿侧擦事件放大惩罚
        swing_conf = torch.sigmoid((self.leg_phase - 0.55) / 0.06)
        stumble_gain = 0.85 + 0.70 * swing_conf

        # 5. 偏严格门控：脚下台阶起伏明确时才强激活，平地尽量静音
        if hasattr(self, "measured_heights"):
            nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
            g = self.measured_heights.view(self.num_envs, nx, ny)
            cx, cy = nx // 2, ny // 2
            underfoot = g[:, max(0, cx - 2):min(nx, cx + 3), max(0, cy - 2):min(ny, cy + 3)]
            underfoot_range = underfoot.amax(dim=(1, 2)) - underfoot.amin(dim=(1, 2))
            center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
            front = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]
            front_delta = torch.abs(front.mean(dim=(1, 2)) - center.mean(dim=(1, 2)))
            # 软门控：脚下起伏 + 前方立面线索联合判定，提前在“台阶前沿”激活
            underfoot_conf = torch.sigmoid((underfoot_range - 0.035) / 0.008)
            front_conf = torch.sigmoid((front_delta - 0.012) / 0.004)
            stair_like_conf = torch.maximum(underfoot_conf, front_conf)
            underfoot_gate = 0.08 + 0.92 * stair_like_conf
        else:
            underfoot_gate = torch.ones(self.num_envs, device=self.device)

        cmd_forward_conf = torch.sigmoid((self.commands[:, 0] - 0.08) / 0.05)
        forward_gate = 0.35 + 0.65 * cmd_forward_conf
        
        return torch.sum(penalty * force_threshold_mask * stumble_gain, dim=1) * underfoot_gate * forward_gate

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
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            height_gate = 0.25 + 0.75 * torch.maximum(stair_conf, 0.6 * cmd_conf)
        else:
            height_gate = 0.25 + 0.75 * cmd_conf
        return height_reward * height_gate

    def _reward_tracking_lin_vel(self):
        """
        线速度跟踪（带角速度一致性门控）
        目的：前进拿分必须建立在“基本服从 yaw 纠偏目标”之上，抑制平地绕圈刷分。
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        yaw_target = self.yaw_track_target if hasattr(self, "yaw_track_target") else self.commands[:, 2]
        yaw_err = torch.abs(yaw_target - self.base_ang_vel[:, 2])
        # 误差越大门控越低；保留 0.2 底座避免训练早期梯度中断
        yaw_align_gate = 0.2 + 0.8 * torch.sigmoid((0.35 - yaw_err) / 0.08)

        forward_mask = (self.commands[:, 0] > 0.1).float()
        gate = forward_mask * yaw_align_gate + (1.0 - forward_mask)

        # 平地抗转圈：在非楼梯区若偏航角速度过大，削弱线速度刷分收益
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        else:
            stair_conf = torch.zeros(self.num_envs, device=self.device)
        flat_conf = 1.0 - stair_conf
        spin_conf = torch.sigmoid((torch.abs(self.base_ang_vel[:, 2]) - 0.30) / 0.10)
        anti_spin_gate = 1.0 - 0.35 * flat_conf * spin_conf * forward_mask

        # 台阶邻域反怠速：临近楼梯却低推进时，线速度得分不能过高
        near_stair_conf = torch.sigmoid((stair_conf - 0.10) / 0.08)
        progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.16) / 0.06)
        anti_stall_gate = 1.0 - 0.55 * near_stair_conf * (1.0 - progress_conf) * forward_mask

        return lin_reward * gate * anti_spin_gate * anti_stall_gate

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
        return torch.where(yaw_err <= e0, near_reward, tail_reward)

    # def _reward_stair_alignment(self):
    #     """
    #     [雷达防逃跑大棒]：基于你原本的逻辑改造
    #     只要有前进指令，眼前必须有台阶，且必须正对台阶！
    #     """
    #     if not hasattr(self, 'measured_heights'):
    #         return torch.zeros(self.num_envs, device=self.device)

    #     # 1. 意图锁：只有被指令要求前进时，才实行连坐惩罚
    #     forward_command_mask = (self.commands[:, 0] > 0.1).float()

    #     # 2. 还原你定义的核心区域高度图
    #     heights = self.measured_heights.view(self.num_envs, 17, 11)
    #     left_h = torch.mean(heights[:, 8:12, :3], dim=(1, 2))
    #     right_h = torch.mean(heights[:, 8:12, -3:], dim=(1, 2))
        
    #     # 3. 计算看歪的程度
    #     perception_skew = torch.abs(left_h - right_h)
        
    #     # 4. 检测前方到底有没有台阶
    #     stair_mask = (torch.max(heights[:, 8:12, :], dim=1)[0].max(dim=1)[0] - 
    #                   torch.min(heights[:, 8:12, :], dim=1)[0].min(dim=1)[0]) > 0.03
        
    #     # 【大棒 A】：歪头惩罚 (只有面前确实是台阶时，歪头才有意义并被惩罚)
    #     crooked_penalty = perception_skew * stair_mask.float()
        
    #     # 【大棒 B】：逃跑惩罚 (眼前根本没台阶？说明你转头向平地跑了！直接拉满惩罚！)
    #     runaway_penalty = (~stair_mask).float() * 1.0  

    #     # [核心修复] 利用局部极差判定是否在真实台阶上
    #     h_max = torch.max(self.measured_heights, dim=1)[0]
    #     h_min = torch.min(self.measured_heights, dim=1)[0]
    #     is_truly_on_stairs = (h_max - h_min > 0.04).float()
        
    #     # 综合大棒，取负号返回
    #     return (crooked_penalty + runaway_penalty) * forward_command_mask * is_truly_on_stairs

    def _reward_vision_stair_drive(self):
        """
        [视觉对齐约束]：全程生效，看到台阶线索时显著增强
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

        h_start = cam_cfg.height // 2
        w_start = cam_cfg.width // 3

        left_view = torch.mean(depth_img[:, h_start:, :w_start], dim=(1, 2))
        right_view = torch.mean(depth_img[:, h_start:, -w_start:], dim=(1, 2))

        # 【视觉看歪惩罚】：左右深度不一样说明侧身走/歪着走
        vision_skew_penalty = torch.abs(left_view - right_view)

        # 地形侧连续置信度（楼梯邻域兜底）
        stair_conf = self._estimate_on_stairs_confidence(update_state=False)
        if hasattr(self, "measured_heights"):
            nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
            g = self.measured_heights.view(self.num_envs, nx, ny)
            cx, cy = nx // 2, ny // 2
            center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
            front = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]
            front_delta = torch.abs(front.mean(dim=(1, 2)) - center.mean(dim=(1, 2)))
            near_stair_conf = torch.sigmoid((front_delta - 0.008) / 0.004)
        else:
            near_stair_conf = torch.zeros_like(stair_conf)
        terrain_conf = torch.maximum(stair_conf, near_stair_conf)

        # 视觉可见性置信度：只要相机“看到前方近距立面/台阶线索”，就提升约束强度
        depth_strip = torch.mean(depth_img[:, h_start:, :], dim=1)
        left_depth = torch.mean(depth_strip[:, :w_start], dim=1)
        center_depth = torch.mean(depth_strip[:, w_start:-w_start], dim=1)
        right_depth = torch.mean(depth_strip[:, -w_start:], dim=1)
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

        total_penalty = 0.55 * vision_skew_penalty + 0.45 * consistency_penalty
        return total_penalty * forward_command_mask * stair_gate

    def _reward_stand_still(self):
        cmd_planar = torch.norm(self.commands[:, :2], dim=1)
        no_cmd_mask = (cmd_planar < 0.05).float()

        idle_lin = torch.norm(self.base_lin_vel[:, :2], dim=1)
        idle_yaw = torch.abs(self.base_ang_vel[:, 2])
        idle_ang_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        idle_lin_pen = torch.clamp((idle_lin - 0.03) / 0.10, min=0.0, max=1.0)
        idle_yaw_pen = torch.clamp((idle_yaw - 0.08) / 0.25, min=0.0, max=1.0)
        idle_ang_xy_pen = torch.clamp((idle_ang_xy - 0.08) / 0.25, min=0.0, max=1.0)
        idle_penalty = idle_lin_pen + 0.5 * idle_yaw_pen + 0.4 * idle_ang_xy_pen

        # 【核心修复】：基于二维平面合成速度评估怠工，兼容倒退和横移
        cmd_active = (cmd_planar > 0.12).float()
        target_v_planar = 0.24 * cmd_planar + 0.02
        
        actual_v_planar = torch.norm(self.base_lin_vel[:, :2], dim=1)
        under_speed = torch.clamp(
            (target_v_planar - actual_v_planar) / (target_v_planar + 0.12),
            min=0.0,
            max=1.0,
        )
        
        # 方向背离惩罚：利用向量点乘判断。如果实际速度方向与指令方向夹角过大（点乘<0）则惩罚
        dot_product = self.commands[:, 0] * self.base_lin_vel[:, 0] + self.commands[:, 1] * self.base_lin_vel[:, 1]
        backtrack_pen = torch.clamp((-dot_product) / 0.12, min=0.0, max=1.0)
        
        low_progress = 0.75 * under_speed + 0.25 * backtrack_pen
        return no_cmd_mask * idle_penalty + cmd_active * low_progress

    def _reward_contact(self):
        """ 奖励左右腿步态交替 """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            stance_hit = (contact & is_stance).float()
            swing_clear = ((~contact) & (~is_stance)).float()
            swing_touch = (contact & (~is_stance)).float()
            stance_miss = ((~contact) & is_stance).float()

            # 支撑相正确触地 + 摆动相干净离地；拖步/失支撑给更重惩罚
            res += 0.70 * stance_hit + 0.45 * swing_clear
            res -= 0.95 * swing_touch
            res -= 0.75 * stance_miss

        # 推进门控：有命令但不推进时，contact 不应成为主要刷分来源
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        cmd_gate = torch.clamp((cmd_speed - 0.06) / 0.32, min=0.0, max=1.0)
        no_cmd_mask = (cmd_speed < 0.06).float()
        progress_conf = torch.sigmoid((self.base_lin_vel[:, 0] - 0.07) / 0.05)
        cmd_mode_gate = 0.08 + 0.92 * cmd_gate * (0.40 + 0.60 * progress_conf)

        idle_lin = torch.norm(self.base_lin_vel[:, :2], dim=1)
        idle_yaw = torch.abs(self.base_ang_vel[:, 2])
        # 0 指令下保留小保底用于姿态微调；一旦漂移/转动，保底自动减弱
        idle_stable_conf = torch.sigmoid((0.04 - idle_lin) / 0.02) * torch.sigmoid((0.10 - idle_yaw) / 0.05)
        idle_floor = 0.012 + 0.028 * idle_stable_conf

        locomotion_gate = (1.0 - no_cmd_mask) * cmd_mode_gate + no_cmd_mask * idle_floor

        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        double_flight = ((~left_contact) & (~right_contact)).float()
        cmd_forward = (self.commands[:, 0] > 0.08).float()
        hop_penalty = double_flight * (0.15 + 0.85 * cmd_forward)

        return res * locomotion_gate - 0.55 * hop_penalty
    
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
