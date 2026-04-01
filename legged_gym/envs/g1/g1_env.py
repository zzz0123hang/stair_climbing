from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import warp as wp
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
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
        self.obs_history_buf = torch.zeros(self.num_envs, 141, device=self.device, dtype=torch.float)
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

    def _estimate_on_stairs_mask(self, env_ids=None, update_state=True):
        """局部 ROI + 双阈值滞回的楼梯判定，减少边缘误判和模式抖动。"""
        if not hasattr(self, "measured_heights"):
            n = self.num_envs if env_ids is None else len(env_ids)
            return torch.zeros(n, dtype=torch.bool, device=self.device)

        h = self.measured_heights if env_ids is None else self.measured_heights[env_ids]
        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        g = h.view(-1, nx, ny)

        cx, cy = nx // 2, ny // 2
        local = g[:, max(0, cx - 3):min(nx, cx + 4), max(0, cy - 2):min(ny, cy + 3)]
        center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
        front = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]

        local_range = local.amax(dim=(1, 2)) - local.amin(dim=(1, 2))
        front_delta = torch.abs(front.mean(dim=(1, 2)) - center.mean(dim=(1, 2)))

        # 双阈值滞回：进入门槛高，退出门槛低，避免切换抖动
        enter = (local_range > 0.05) | (front_delta > 0.02)
        exit_ = (local_range < 0.03) & (front_delta < 0.01)

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
        if not hasattr(self, "measured_heights"):
            n = self.num_envs if env_ids is None else len(env_ids)
            return torch.zeros(n, dtype=torch.float, device=self.device)

        h = self.measured_heights if env_ids is None else self.measured_heights[env_ids]
        nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
        g = h.view(-1, nx, ny)

        cx, cy = nx // 2, ny // 2
        local = g[:, max(0, cx - 3):min(nx, cx + 4), max(0, cy - 2):min(ny, cy + 3)]
        center = g[:, max(0, cx - 1):min(nx, cx + 2), max(0, cy - 1):min(ny, cy + 2)]
        front = g[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)]

        local_range = local.amax(dim=(1, 2)) - local.amin(dim=(1, 2))
        front_delta = torch.abs(front.mean(dim=(1, 2)) - center.mean(dim=(1, 2)))

        # 用 sigmoid 替代硬阈值：在阈值附近保持可学习的平滑过渡
        range_conf = torch.sigmoid((local_range - 0.04) / 0.006)
        front_conf = torch.sigmoid((front_delta - 0.015) / 0.004)
        stair_conf = torch.maximum(range_conf, front_conf)

        # 时间平滑，减少帧间抖动
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

        # 双阶段课程：从第3级开始拉起后期项，满级时拉满（按 raw difficulty 计算）
        late_stage_start_level = 3.0
        late_stage_start_raw = late_stage_start_level / rows
        late_stage_den = max(terrain_progress_max - late_stage_start_raw, 1e-6)
        late_stage_progress = torch.clamp(
            (terrain_progress_raw - late_stage_start_raw) / late_stage_den,
            min=0.0,
            max=1.0
        )
        dt = self.dt

        def safe_update(name, val):
            if name in self.reward_scales:
                self.reward_scales[name] = val * dt

        # 2. 【前半程：导航与跟踪优先】
        # 线速度跟踪后期退火，避免在高难阶段“稳走拿分”压制楼梯专项目标
        safe_update("tracking_lin_vel", 2.0 - 0.5 * late_stage_progress)
        # 角速度跟踪尽早拉起，保证视觉纠偏命令有执行动力
        safe_update("tracking_ang_vel", 1.6 + 0.6 * terrain_progress)
        # 前置引导：仅在前期保留适中牵引，后期逐步退到 0，避免与楼梯内精修目标冲突
        safe_update("approach_stairs", 0.30 * (1.0 - late_stage_progress))
        # contact 不宜过早拉高，否则会压制导航目标；后期上限进一步下调到 0.8
        safe_update("contact", 0.4 + 0.4 * late_stage_progress)
 
        # 3. 【姿态课程】中后期加强稳定性
        self.dynamic_base_height_target = 0.72 + (0.03 * terrain_progress)
        safe_update("pelvis_height", 0.1 + 0.2 * terrain_progress)
        safe_update("orientation", -0.3 - 0.7 * late_stage_progress)
        safe_update("feet_air_time", 1.0 + 1.0 * terrain_progress)

        # 4. 【后半程：楼梯安全精修】
        safe_update("feet_stumble", -0.3 - 0.7 * late_stage_progress)
        safe_update("foot_flatness", -0.05 - 0.35 * late_stage_progress)
        # 安全贴合：后期拉到约 -0.6，并采用缓升避免前期过猛
        safe_update("foot_support_rect", -0.08 - 0.52 * torch.pow(late_stage_progress, 1.5))
        safe_update("stair_clearance", -0.2 - 0.8 * late_stage_progress)
        # 交替步态：后期上限提高到约 -0.9，强化“非同级双脚支撑”
        safe_update("stair_alternation", -0.15 - 0.85 * late_stage_progress)

        # 5. 【运动质量】全程稳态约束，后期略强化
        # 随台阶难度提升，放松 z 轴速度约束，允许更积极抬升越阶
        safe_update("lin_vel_z", -0.4 + 0.2 * late_stage_progress)
        # 后期允许上下跃迁，但横向抖动和高频抖腿应更受控
        safe_update("action_rate", -0.02 - 0.01 * late_stage_progress)
        safe_update("ang_vel_xy", -0.05 - 0.02 * late_stage_progress)

        self.current_terrain_progress = terrain_progress
        self.current_late_stage_progress = late_stage_progress
        
        # 【核心修正】：直接塞进全局 extras，不依赖机器人的 reset，每一步都实时更新！
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["Metrics/metrics_terrain_progress_raw"] = torch.mean(terrain_progress_raw).item()
        self.extras["Metrics/metrics_terrain_progress"] = torch.mean(terrain_progress).item()
        self.extras["Metrics/metrics_late_stage_progress"] = torch.mean(late_stage_progress).item()
        return super().compute_reward()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.update_feet_state()

        # 1. 测高雷达更新
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            
        # 2. 视觉老司机：每一步物理仿真后，重新渲染图像并进行方向盘纠偏
        if hasattr(self, 'stereo_cam') and self.cfg.sensor.stereo_cam.enable:
            self._update_warp_camera()

            # 前进控制权软门控：避免 x 指令在阈值附近抖动导致“接管/不接管”硬切
            forward_cmd = self.commands[:, 0]
            forward_conf = torch.sigmoid((forward_cmd - 0.10) / 0.05)  # 0~1

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

            # d. 【动态几何探角 - 软判定】
            # 将“是否需要逃逸”从硬开关改为连续置信度，减少阈值抖动
            ridge_ratio = torch.minimum(
                left_depth / (center_depth + 1e-6),
                right_depth / (center_depth + 1e-6)
            )
            valley_ratio = center_depth / (torch.maximum(left_depth, right_depth) + 1e-6)
            ridge_conf = torch.sigmoid((ridge_ratio - 1.55) / 0.08)
            valley_conf = torch.sigmoid((valley_ratio - 1.08) / 0.05)
            escape_conf = torch.maximum(ridge_conf, valley_conf)  # 0~1

            # e. 【正常对齐逻辑：寻找平衡】
            # 右浅左深（朝左歪）-> 右转 (-yaw) 纠偏
            depth_diff = right_depth - left_depth
            # align_err = left_depth - right_depth
            # 对齐死区软化：阈值附近不再硬清零，而是渐进减弱纠偏
            align_gain = torch.sigmoid((torch.abs(depth_diff) - 0.10) / 0.03)
            align_err = depth_diff * align_gain

            # f. 【极端死角逃生逻辑 - 融合版】
            # 遇到 90 度尖角/深坑时触发，哪边开阔往哪边强行拽回台阶面
            # 但同时融合 align_err 的强度信息，避免过度反应
            escape_direction = torch.where(left_depth > right_depth,
                                          torch.full_like(align_err, 1.0),
                                          torch.full_like(align_err, -1.0))
            # 逃生角速度软幅值：取消过大的硬下限，避免轻微误判时“猛打方向”
            center_gap = torch.clamp(center_depth - 0.5 * (left_depth + right_depth), min=0.0)
            escape_strength = torch.clamp(torch.abs(depth_diff) * 2.0 + center_gap * 0.6, min=0.0, max=1.2)
            escape_yaw = escape_direction * escape_strength

            # 方向盘指令融合
            raw_yaw = (1.0 - escape_conf) * align_err + escape_conf * escape_yaw

            # g. 【纯视觉 Sim2Real 修复：封堵侧身走”长廊错觉”】
            # 彻底抛弃上帝视角 (self.terrain_types == 2)！
            # 直接利用机器人的局部测高雷达 (大范围高程图) 判断所处地形。
            stair_conf = self._estimate_on_stairs_confidence(update_state=True)

            # 👑【动态控制权移交 - 软权重版】
            # 开阔且非楼梯时逐步放手；保留最多 60% 放手，避免纠偏被完全压掉
            override_w = 0.6 * torch.pow(open_conf, 1.5) * (1.0 - stair_conf)
            desired_yaw = (1.0 - override_w) * raw_yaw

            # 限制极速，防止打方向盘过猛摔倒 (建议保持 1.0，保证尖角逃生速度)
            target_yaw_vel = torch.clamp(desired_yaw, -1.0, 1.0)
            alpha = 0.2
            smoothed_yaw_vel = (1.0 - alpha) * self.commands[:, 2] + alpha * target_yaw_vel
            smoothed_yaw_vel = torch.where(torch.abs(smoothed_yaw_vel) < 0.03,
                                           torch.zeros_like(smoothed_yaw_vel),
                                           smoothed_yaw_vel)

            # h. 【执行指令覆盖】
            if self.cfg.commands.heading_command:
                # Heading 模式：按前进置信度渐进接管，避免指令阈值附近硬切
                self.commands[:, 3] = self.commands[:, 3] + forward_conf * smoothed_yaw_vel * self.dt
            else:
                # 角速度模式下，直接接管并覆盖底层随机角速度
                # 非前进状态下做指数回零，避免历史 yaw 残留导致持续转圈
                idle_decay = 0.85
                idle_yaw = self.commands[:, 2] * idle_decay
                idle_yaw = torch.where(torch.abs(idle_yaw) < 0.02,
                                       torch.zeros_like(idle_yaw),
                                       idle_yaw)
                blended_yaw = forward_conf * smoothed_yaw_vel + (1.0 - forward_conf) * idle_yaw
                self.commands[:, 2] = torch.clamp(blended_yaw, -1.0, 1.0)

        # print(f"前进速度：{forward_mask}")
        # print(f"[纠偏] yaw_vel: {target_yaw_vel.mean().item():.3f}, base_ang: {self.base_ang_vel[:, 2].mean().item():.3f}, cmd: {self.commands[:, 2].mean().item():.3f}")
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
        组装学生 (781维) 和 老师 (971维) 的非对称观测向量 
        学生 = 141维历史本体感受 + 640维压缩视觉特征
        老师 = 学生观测 + 187维地形高度采样 + 3维物理特权参数 (摩擦、负载、真速)
        """
        # 1. 计算当前时间相位 (2 维: sin/cos)
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # 2. 构造当前帧的本体感受数据 (47 维)
        # 顺序必须固定：角速度(3), 重力投影(3), 指令(3), 关节位置(12), 关节速度(12), 上一次动作(12), 相位(2)
        current_proprio = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,           # 3
            self.projected_gravity,                                # 3
            self.commands[:, :3] * self.commands_scale,            # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel,                # 12
            self.actions,                                          # 12
            sin_phase,                                             # 1
            cos_phase                                              # 1
        ), dim=-1)

        # 3. 更新历史堆叠缓冲区 (47 * 3 = 141 维)
        # 如果开启噪声，仅对当前帧加噪，防止历史数据噪声累积
        if self.add_noise:
            current_proprio += (2 * torch.rand_like(current_proprio) - 1) * self.noise_scale_vec[:47]
        
        # 将新数据推入，舍弃最旧的一帧
        self.obs_history_buf = torch.cat((current_proprio, self.obs_history_buf[:, :-47]), dim=-1)

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
    
    # 请放在 g1_env.py 的 G1Robot 类中
    def _resample_commands(self, env_ids):
        """
        [终极修复版] 指令重采样与软牵引课程
        """
        # 1. 基础采样：调用基类生成包含前后左右、自转的彻底随机全向指令
        super()._resample_commands(env_ids)

        # 2. 【概率软牵引课程】 (仅在三维地形上生效)
        if self.cfg.terrain.mesh_type == 'trimesh':
            # max_level = self.cfg.terrain.num_rows - 1

            # 重采样时先刷新这批环境的高度，避免使用上一位置的旧测高
            if self.cfg.terrain.measure_heights and len(env_ids) > 0:
                self.measured_heights[env_ids] = self._get_heights(env_ids)

            on_stairs_mask = self._estimate_on_stairs_mask(env_ids=env_ids, update_state=True)
            on_stairs_mask = on_stairs_mask | self.cfg.env.test

            # rand_probs = torch.rand(len(env_ids), device=self.device)

            # 【楼梯牵引】：100% 在楼梯上的机器人被正向牵引（视觉纠偏已恢复，不再需要随机探索）
            traction_mask = on_stairs_mask

            if traction_mask.any():
                # a. 强制下达向前的指令 (0.5 ~ max m/s)，逼迫它们往前走
                self.commands[env_ids[traction_mask], 0] = torch_rand_float(
                    0.5, self.command_ranges["lin_vel_x"][1],
                    (traction_mask.sum().item(), 1), device=self.device
                ).squeeze(1)

                # b. 收紧侧移 (乘以 0.1)，防止在爬楼梯时走出夸张的”螃蟹步”导致轻易绊倒
                self.commands[env_ids[traction_mask], 1] *= 0.1

                # c. 处理旋转指令
                if self.cfg.commands.heading_command:
                    # Heading 模式下，锁定为当前朝向，让视觉老司机平滑微调
                    from isaacgym.torch_utils import quat_apply
                    forward = quat_apply(self.base_quat[env_ids[traction_mask]], self.forward_vec[env_ids[traction_mask]])
                    current_heading = torch.atan2(forward[:, 1], forward[:, 0])
                    self.commands[env_ids[traction_mask], 3] = current_heading
                else:
                    self.commands[env_ids[traction_mask], 2] = 0.0

            # 【平地上：90%前向牵引，10%保留随机探索】
            flat_mask = ~on_stairs_mask
            if flat_mask.any():
                flat_env_ids = env_ids[flat_mask]
                flat_rand_probs = torch.rand(len(flat_env_ids), device=self.device)
                # 90% 前向牵引：给定更高下限的前向速度，减少无效慢挪
                forward_mask_flat = flat_rand_probs < 0.9
                if forward_mask_flat.any():
                    min_forward = min(0.4, float(self.command_ranges["lin_vel_x"][1]))
                    self.commands[flat_env_ids[forward_mask_flat], 0] = torch_rand_float(
                        min_forward, self.command_ranges["lin_vel_x"][1],
                        (forward_mask_flat.sum().item(), 1), device=self.device
                    ).squeeze(1)
                # 平地统一收紧侧移，避免横摆抵消净前进距离
                self.commands[flat_env_ids, 1] *= 0.2
                # 10% 保持_resample_commands 的随机 yaw/x 细节（仅侧移已收紧）
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

    def _get_terrain_heights(self, env_ids, x, y):
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

        # 3. 转换坐标
        x_grid = (x + border) / horizontal_scale
        y_grid = (y + border) / horizontal_scale  # 注意：这里假设 vertical_scale 和 horizontal_scale 相同，或者这是正方形网格

        # 4. [核心修复] 使用真实的维度上限进行 Clip
        # x 对应第 0 维 (rows), y 对应第 1 维 (cols)
        x_grid_idx = torch.clip(x_grid.long(), 0, num_rows - 1)

        # --- 修正点 ---
        # 错误写法: y_grid_idx = torch.clip(y_grid_idx, 0, num_cols - 1)
        # 正确写法: 应该使用 y_grid.long() 作为输入
        y_grid_idx = torch.clip(y_grid.long(), 0, num_cols - 1)

        # 5. 查表
        heights = self.height_samples[x_grid_idx, y_grid_idx] * vertical_scale
        return heights

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
        if hasattr(self, 'on_stairs_state'):
            self.on_stairs_state[env_ids] = False
        if hasattr(self, 'on_stairs_conf_state'):
            self.on_stairs_conf_state[env_ids] = 0.0
        # 必须调用父类的 reset_idx，否则机器人摔倒后不会重置！
        super().reset_idx(env_ids)

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
        front = heights[:, max(0, cx + 2):min(nx, cx + 6), max(0, cy - 1):min(ny, cy + 2)].mean(dim=(1, 2))

        # 前方相对脚下的正向高度差（仅关心“前高后低”的上台阶趋势）
        delta_h = torch.clamp(front - center, min=0.0, max=0.25)

        # 与 terrain.py / stair_alternation 保持口径一致：difficulty = level / num_rows
        rows = float(max(self.cfg.terrain.num_rows, 1))
        difficulty = self.terrain_levels.float() / rows
        target_step_height = 0.05 + 0.1 * difficulty

        # 高度匹配奖励：越接近目标台阶高度越高（避免“只要有高度差就刷分”）
        sigma = 0.02
        height_match = torch.exp(-torch.square(delta_h - target_step_height) / (2.0 * sigma * sigma))
        near_stair = (delta_h > 0.01).float()

        on_stairs = self._estimate_on_stairs_mask(update_state=False).float()
        forward_cmd = (self.commands[:, 0] > 0.1).float()
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0, max=1.0)
        # 只在非楼梯区域生效：专注解决“走到台阶前”的问题
        gate = (1.0 - on_stairs)

        return height_match * near_stair * forward_cmd * forward_vel * gate
    
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
        underfoot_mask = underfoot_range > 0.04
        
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
        support_gate = torch.sigmoid((min_fz - 5.0) / 3.0)
        
        # 6. 严格前进门控
        forward_mask = self.commands[:, 0] > 0.1
        
        # 7. 误差惩罚 + 匹配奖励（函数仍返回“可正可负”的信号）
        # 说明：该项在 config 中配负权重。
        # - 误差大 -> 函数为正 -> 乘负权重后是惩罚
        # - 误差小 -> 函数可为负 -> 乘负权重后是正奖励（匹配即正奖）
        norm_error = diff_error / (target_step_height + 1e-4)
        penalty_term = torch.clamp(norm_error, min=0.0, max=2.0)

        sigma = 0.20
        match_reward = torch.exp(-torch.square(norm_error) / (2.0 * sigma * sigma))
        bonus_weight = 1.10
        common_gate = underfoot_mask.float() * forward_mask.float()
        penalty_gate = common_gate * hard_double_stance.float() * hard_contact.float()
        bonus_gate = common_gate * soft_double_stance * support_gate

        return penalty_term * penalty_gate - bonus_weight * match_reward * bonus_gate

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
        
        # 摆动相且确认没踩实的情况：保留 0.06 的死区
        is_pure_swing = (1.0 - is_stance_phase) * (~is_firm_contact).float()
        swing_penalty = torch.clamp(horizontal_deviation - 0.2, min=0.0) * is_pure_swing
        
        # 4. 软门控：全程保留脚掌姿态约束，楼梯段适度增强
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            flatness_gate = 0.8 + 0.4 * stair_conf  # [0.8, 1.2]
        else:
            flatness_gate = torch.ones(self.num_envs, device=self.device)

        return torch.sum(stance_penalty + swing_penalty, dim=1) * flatness_gate

    def _reward_foot_support_rect(self):
        """
        [专家强化版] 针对大落差台阶优化的踩空惩罚：
        1. 引入 0.05m 物理死区，过滤地形微小噪声。
        2. 保留摆动末期 (90% phase) 预警，但采用线性增压防止过度惩罚。
        3. 结合支撑力反馈，确保“踩得越重，惩罚越严”。
        """
        # --- 1. 基础参数配置 ---
        progress = getattr(self, 'current_late_stage_progress',
                           getattr(self, 'current_terrain_progress', 0.0))
        if isinstance(progress, torch.Tensor):
            progress = progress.unsqueeze(1)
        
        # 动态边缘死区：从 0.05m 收缩到 0.015m
        # 早期让它敢于落脚，后期逼迫它实现 1.5cm 的极限贴合安全步态
        edge_threshold = 0.05 - 0.03 * progress
        
        # 2. 定义脚掌四个角点的局部坐标 (前、后、侧向偏移)
        dist_front, dist_back, dist_side = 0.12, -0.05, 0.03
        corners_local = torch.tensor([
            [dist_front,  dist_side, 0.0], [dist_front, -dist_side, 0.0],
            [dist_back,   dist_side, 0.0], [dist_back,  -dist_side, 0.0]
        ], device=self.device, dtype=torch.float32)

        # 3. 空间几何变换：将角点转到世界坐标系
        feet_pos = self.feet_pos 
        feet_rot = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        
        # 广播并应用四元数旋转
        feet_rot_exp = feet_rot.unsqueeze(2).repeat(1, 1, 4, 1)
        corners_exp = corners_local.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.feet_num, 1, 1)
        
        from isaacgym.torch_utils import quat_apply
        corners_world = quat_apply(feet_rot_exp.view(-1, 4), corners_exp.view(-1, 3)).view(self.num_envs, self.feet_num, 4, 3)
        corners_world += feet_pos.unsqueeze(2)

        # 4. 地形高度采样
        x_query, y_query = corners_world[..., 0].view(-1), corners_world[..., 1].view(-1)
        # 从高度图中获取四个角点对应的地形高度
        terrain_heights = self._get_terrain_heights(None, x_query, y_query).view(self.num_envs, self.feet_num, 4)
        
        # 5. 【核心计算】：计算脚下地形极差
        h_range = torch.max(terrain_heights, dim=-1)[0] - torch.min(terrain_heights, dim=-1)[0]

        cliff_error = torch.clamp(h_range - edge_threshold, min=0.0)
        
        # 2. 只有当 cliff_error 大于 0 时，才开始计算惩罚。
        # 当 cliff_error = 0 时，1 - exp(0) = 1 - 1 = 0.0 (真正的 0 惩罚！)
        cliff_penalty = 1.0 - torch.exp(-cliff_error / 0.01)
        
        # --- 6. 【动态掩码逻辑】：响应你的预警需求 ---
        # a. 支撑相 (Phase < 0.55)：全额激活惩罚 (1.0)
        is_stance = (self.leg_phase < 0.55).float()
        
        # b. 摆动末期预警 (Phase 0.90 ~ 1.0)：线性增加权重 (0.0 -> 0.5)
        # 这保证了机器人有“预防意识”，但不会因为空中的微小姿态偏差被处以极刑
        is_pre_land = torch.clamp((self.leg_phase - 0.90) / 0.10, min=0.0, max=1.0) * 0.5
        
        # 融合后的动态激活掩码
        dynamic_active_mask = torch.max(is_stance, is_pre_land)
        
        # 7. 【压力反馈】：踩实后的惩罚增压
        # 只有真正承重时 (50N以上)，才认为踩空是极其严重的
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_weight = torch.clamp(contact_force_z / 50.0, 0.0, 1.0)
        
        # 8. 半严格门控：仅在脚下确实进入台阶态时强化该项，平地尽量静音
        if hasattr(self, "measured_heights"):
            nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
            g = self.measured_heights.view(self.num_envs, nx, ny)
            cx, cy = nx // 2, ny // 2
            underfoot = g[:, max(0, cx - 2):min(nx, cx + 3), max(0, cy - 2):min(ny, cy + 3)]
            underfoot_range = underfoot.amax(dim=(1, 2)) - underfoot.amin(dim=(1, 2))
            # 半严格：脚下起伏>4cm 视为楼梯区；保留 0.1 底值防止边界瞬断
            underfoot_gate = 0.1 + 0.9 * (underfoot_range > 0.04).float()
        else:
            underfoot_gate = torch.ones(self.num_envs, device=self.device)

        # 9. 汇总：(边缘判定) * (动态预警掩码) * (物理压力权重) * (脚下台阶门控)
        # 在空中靠 dynamic_active_mask 引导，落地后靠 contact_weight 压实惩罚
        total_penalty = cliff_penalty * dynamic_active_mask * (1.0 + contact_weight) * underfoot_gate.unsqueeze(1)
        
        return torch.sum(total_penalty, dim=1)

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
        if hasattr(self, "measured_heights"):
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
        # ==========================================================
        
        # --- 3. 实时采样原始地形高度 (带有突变风险的 Raw Data) ---
        raw_start_h = self._get_terrain_heights(None, self.feet_pos[..., 0].view(-1), self.feet_pos[..., 1].view(-1)).view(self.num_envs, -1)
        raw_end_h = self._get_terrain_heights(None, predicted_land_x.view(-1), predicted_land_y.view(-1)).view(self.num_envs, -1)
        
        # ==========================================================
        # 👑 4. 【核心创新：双端高度锁存器 (Double-ended Height Latch)】
        # ==========================================================
        # 初始化 PyTorch 缓存变量 (只在第一次调用时创建)
        if not hasattr(self, 'cached_start_h'):
            self.cached_start_h = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            self.cached_end_h = torch.zeros((self.num_envs, self.feet_num), device=self.device)
            
        # 判断当前腿是否踩在地上 (t == 0.0)
        is_stance = (t == 0.0)
        
        # 魔法发生的地方：
        # 如果脚踩在地上 (is_stance=True)，不断用最新的地形高度刷新缓存。
        # 一旦脚起飞 (is_stance=False)，立即停止刷新，死死锁住起飞那一瞬间的高度！
        self.cached_start_h = torch.where(is_stance, raw_start_h, self.cached_start_h)
        self.cached_end_h = torch.where(is_stance, raw_end_h, self.cached_end_h)
        
        # 提取绝对稳定、无突变的起点和终点高度！
        current_terrain_h = self.cached_start_h
        future_terrain_h = self.cached_end_h
        # ==========================================================
        
        # --- 5. 基线插值轨迹规划 ---
        # 现在这条线，宛如焊死在空间中的钢筋，绝对不会在半空中发生任何抖动或突变！
        linear_h = current_terrain_h * (1.0 - t) + future_terrain_h * t
        
        cmd_vel_norm = torch.clamp(torch.norm(self.commands[:, :2], dim=1), max=1.0).unsqueeze(1)
        
        # ================== 地形感知动态解耦 ==================
        step_diff = future_terrain_h - current_terrain_h
        
        # 仅在上坡/上台阶时提取正向高度差，平地和下坡为 0
        step_up_diff = torch.clamp(step_diff, min=0.0) 
        
        # 完美的数学振幅：基础余量 + 0.5 * 上坡高度差
        sine_amplitude = 0.03 + 0.05 * cmd_vel_norm + 0.5 * step_up_diff
        
        # 生成正弦波 (下坡时 step_up_diff 为 0，仅有 3cm 保底，平滑下探)
        sine_lift = sine_amplitude * torch.sin(t * np.pi)
        
        target_h = linear_h + sine_lift
        # ====================================================
        
        # --- 7. 计算高度误差惩罚 ---
        current_foot_h = self.feet_pos[..., 2]
        error = torch.clamp(target_h - current_foot_h, min=0.0)
        
        penalty = 1.0 - torch.exp(-error / 0.06)

        # 软门控：平地保留基础抬腿约束，临近/处于楼梯时适度增强
        if hasattr(self, "measured_heights"):
            stair_conf = self._estimate_on_stairs_confidence(update_state=False)
            clearance_gate = 0.8 + 0.4 * stair_conf  # [0.8, 1.2]
        else:
            clearance_gate = torch.ones(self.num_envs, device=self.device)
        
        return torch.sum(penalty * is_swinging * clearance_gate.unsqueeze(1), dim=1)

    def _reward_feet_stumble(self):
        """
        [力矢量分析版] 监控水平力/垂直力比例，精准识别“踢台阶”动作。
        """
        # 1. 提取足端力 (假设 self.feet_indices 已经包含所有脚的索引)
        forces = self.contact_forces[:, self.feet_indices, :]
        force_xy = torch.norm(forces[:, :, :2], dim=2) 
        force_z = torch.abs(forces[:, :, 2])
        
        # 2. 计算绊倒比例，+5.0 避免除零，同时过滤掉极其微小的力噪声
        stumble_ratio = force_xy / (force_z + 15.0)
        
        # 3. 构造惩罚映射：水平力 > 垂直力 + 阈值 1.0 时触发
        # 使用 tanh 确保单步惩罚有上限值 1.0
        penalty = torch.tanh(torch.clip(stumble_ratio - 1.5, min=0.0))
        
        # 4. 物理过滤：只有总受力大于 10N 产生的绊倒比例才计入惩罚
        total_force_norm = torch.norm(forces, dim=2)
        force_threshold_mask = (total_force_norm > 10.0).float()

        # 5. 偏严格门控：脚下台阶起伏明确时才强激活，平地尽量静音
        if hasattr(self, "measured_heights"):
            nx, ny = self.measured_grid_shape if hasattr(self, "measured_grid_shape") else (17, 11)
            g = self.measured_heights.view(self.num_envs, nx, ny)
            cx, cy = nx // 2, ny // 2
            underfoot = g[:, max(0, cx - 2):min(nx, cx + 3), max(0, cy - 2):min(ny, cy + 3)]
            underfoot_range = underfoot.amax(dim=(1, 2)) - underfoot.amin(dim=(1, 2))
            # 平地保留 0.1，楼梯区提升到 1.0
            underfoot_gate = 0.1 + 0.9 * (underfoot_range > 0.04).float()
        else:
            underfoot_gate = torch.ones(self.num_envs, device=self.device)
        
        return torch.sum(penalty * force_threshold_mask, dim=1) * underfoot_gate

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
        
        # 4. 允许 3cm 的微小死区
        error = torch.clamp(error - 0.05, min=0.0)
        return torch.exp(-torch.square(error) / 0.01)

    def _reward_tracking_ang_vel(self):
        """
        [角速度跟踪修正版]
        目标：
        1. 小误差区保持高精度（沿用高斯形状）。
        2. 大误差区保留梯度并保持单调下降，避免“误差更大反而加分”。
        """
        yaw_err = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])

        # 分段连续单调：e<=e0 用高斯；e>e0 用“从 e0 奖励值出发”的重尾
        e0 = 0.5
        near_reward = torch.exp(-torch.square(yaw_err) / self.cfg.rewards.tracking_sigma)
        r0 = math.exp(-(e0 * e0) / self.cfg.rewards.tracking_sigma)
        tail_reward = r0 / (1.0 + torch.square((yaw_err - e0) / 0.8))
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
        [视觉防逃跑大棒]：左右深度差惩罚
        只有当机器人真正在楼梯上时，才惩罚左右深度不对称（侧身走/歪着走）
        移除了不合理的 lost_target_penalty（机器人出生时前方1.5m才有台阶，不该惩罚）
        """
        if not hasattr(self, 'visual_obs_buf'):
            return torch.zeros(self.num_envs, device=self.device)

        forward_command_mask = (self.commands[:, 0] > 0.1).float()

        cam_cfg = self.cfg.sensor.stereo_cam
        depth_img = torch.clamp(self.visual_obs_buf.view(self.num_envs, cam_cfg.height, cam_cfg.width), 0.1, 2.0)

        h_start = cam_cfg.height // 2
        w_start = cam_cfg.width // 3

        left_view = torch.mean(depth_img[:, h_start:, :w_start], dim=(1, 2))
        right_view = torch.mean(depth_img[:, h_start:, -w_start:], dim=(1, 2))

        # 【视觉看歪惩罚】：左右深度不一样说明侧身走/歪着走
        vision_skew_penalty = torch.abs(left_view - right_view)

        # [核心验证] 使用统一楼梯判定，避免控制与奖励门控不一致
        is_truly_on_stairs = self._estimate_on_stairs_mask(update_state=False).float()

        return vision_skew_penalty * forward_command_mask * is_truly_on_stairs


    def _reward_contact(self):
        """ 奖励左右腿步态交替 """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            # 逻辑等价判定：如果触地状态符合相位要求，给正分
            res += (contact == is_stance).float()
            res -= (contact & ~is_stance).float()
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
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
        yaw_tolerance = 0.05 + 0.3 * yaw_vel_command.unsqueeze(1)
            
        error_yaw = torch.clamp(torch.abs(error[:, [0, 6]]) - yaw_tolerance, min=0.0)
        
        # 惩罚全部使用平方，保证平滑的梯度
        penalty_roll = torch.sum(torch.square(error_roll) * 1.0, dim=1)
        penalty_pitch = torch.sum(torch.square(error_pitch) * 1.0, dim=1) 
        penalty_yaw = torch.sum(torch.square(error_yaw) * 1.0, dim=1)
        
        return penalty_roll + penalty_pitch + penalty_yaw
