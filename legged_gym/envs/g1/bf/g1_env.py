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
    def init_warp_stereo_camera(self):
        """ 初始化 Warp 双目相机，分配显存并绑定内存地址 """
        if not hasattr(self.cfg, 'sensor') or not self.cfg.sensor.stereo_cam.enable:
            return   
        cam_cfg = self.cfg.sensor.stereo_cam
        self.enable_camera_debug = (self.num_envs == 1)
        # 1. 初始化 Warp
        try:
            if not wp.is_initialized(): wp.init()
            wp.config.device = f"cuda:{self.device_id}" if hasattr(self, 'device_id') else "cuda:0"
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
        down_vec = torch.cross(look_vec, right_vec)
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
        self._init_foot()
        self.init_envs()
        self.session_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.foothold_log = []

    def init_envs(self):
        # 1. 平面地形无需修正
        if self.cfg.terrain.mesh_type == 'plane':
            return
        if not hasattr(self, 'terrain') or self.terrain is None:
            return
        vertical_scale = self.cfg.terrain.vertical_scale
        horizontal_scale = self.cfg.terrain.horizontal_scale
        border_size = self.cfg.terrain.border_size      
        # 3. 获取配置中的偏移量 [12, 4, 0.8]
        init_pos = self.cfg.init_state.pos
        offset_x = init_pos[0]
        offset_y = init_pos[1]
        spawn_offset_z = init_pos[2] # 0.8m
        # 4. [关键修正] 计算采样点的真实世界坐标
        # 机器人实际位置 = 环境原点 + 配置的偏移量
        total_x = self.env_origins[:, 0] + offset_x
        total_y = self.env_origins[:, 1] + offset_y
        # 5. 转换为网格坐标 (Grid Indices)
        grid_x = (total_x + border_size) / horizontal_scale
        grid_y = (total_y + border_size) / horizontal_scale   
        # 取整并限制在地图范围内
        grid_x = grid_x.long().clamp(0, self.terrain.tot_rows - 1)
        grid_y = grid_y.long().clamp(0, self.terrain.tot_cols - 1)   
        # 6. 查表获取脚下的真实地形高度
        raw_heights = self.terrain.height_field_raw[grid_x.cpu().numpy(), grid_y.cpu().numpy()]
        terrain_heights = torch.tensor(raw_heights, device=self.device) * vertical_scale
        self.env_origins[:, 2] = terrain_heights    
        # 8. 更新当前的 root_states (修正第一帧)
        self.root_states[:, 2] = terrain_heights + spawn_offset_z    
        # 9. 刷新物理引擎
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))    
        print(f"[G1 Info] 出生点已修正: 坐标({offset_x}, {offset_y}) 处地形高度 + {spawn_offset_z}m")
    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        # 每一步物理仿真后，重新渲染图像
        if hasattr(self, 'stereo_cam') and self.cfg.sensor.stereo_cam.enable:
            self._update_warp_camera()
        # ---------------------------
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        # [新增] 记录落足瞬间的坐标
        if not hasattr(self, 'foothold_log'):
            self.foothold_log = [] # 暂时存在内存里，或者你可以定期写入文件

        # 检测从摆动到支撑的瞬间 (Touch down)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        touchdown = contact & (~self.last_contacts) # 上一帧没踩，这一帧踩了
        
        if touchdown.any():
            env_ids, foot_ids = torch.where(touchdown)
            for i in range(len(env_ids)):
                e_idx = env_ids[i]
                f_idx = foot_ids[i]
                # 记录此时脚的世界坐标 (相对于台阶中心的相对位置更好，这里先存绝对位置)
                pos = self.feet_pos[e_idx, f_idx].cpu().numpy()
                # 存入列表：[环境ID, 步数, 坐标x, 坐标y, 坐标z]
                self.foothold_log.append([e_idx.item(), self.common_step_counter, pos[0], pos[1], pos[2]])

        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations """
        # 1. 计算相位信息
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)

        # 2. 计算基础本体感知观测 (Proprioception) - 此时长度为 47
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        
        # 3. 计算特权观测
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)

        # 4. 【修复】添加本体感知噪声
        if self.add_noise:
            # 动态匹配维度，防止 obs_buf 长度变化导致广播错误
            current_dim = self.obs_buf.shape[1] 
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:current_dim]

        # 5. 拼接视觉观测 (Visual Observation)
        if hasattr(self.cfg, 'sensor') and self.cfg.sensor.stereo_cam.enable:
            
            if hasattr(self, 'visual_obs_buf'):
                visual_obs = self.visual_obs_buf
                # 归一化处理 (0 ~ 1)
                max_range = self.cfg.sensor.stereo_cam.max_range
                visual_obs = torch.clamp(visual_obs, 0.0, max_range) / max_range
            else:
                # Fallback: 给个零向量防止 Crash
                dim = getattr(self, 'visual_feature_dim', 160)
                visual_obs = torch.zeros((self.num_envs, dim), device=self.device)

            self.obs_buf = torch.cat((self.obs_buf, visual_obs), dim=-1)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, visual_obs), dim=-1)

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
    
    def _debug_visualize_training(self, depth_tensor):
        """ [清爽版] 仅显示深度伪彩图和极值 
            💚 鲜绿色 (Green)
            含义：无效值 / 盲区 / 0米。
            来源：代码中的 colored[mask_invalid] = [0, 255, 0]。
            解释：这些像素的深度值小于 0.001（接近 0）。通常是因为物体贴得太近（小于相机的最小感测距离），或者是由于遮挡、纹理缺失导致的深度丢失（Dropout）。
            💜 深紫色 / 黑色 (Dark Purple / Black)
            含义：距离非常近（接近有效测量的最小值）。
            解释：INFERNO 色谱的起始端（数值 0）是深紫色或黑色。
            🔴 红色 / 橙色 (Red / Orange)
            含义：中等距离。
            解释：随着距离增加，颜色会逐渐变暖，从紫变红再变橙。
            💛 明黄色 / 白色 (Yellow / White)
            含义：距离最远（接近 max_range，比如 3.0 米）。
            解释：INFERNO 色谱的末端（数值 255）是高亮的黄色。如果看到大片黄色，说明那里已经达到或超过了你设定的最大探测范围。
        """
        try:
            # 1. 获取最大量程
            max_r = getattr(self.cfg.sensor.stereo_cam, 'max_range', 3.0)

            # 2. 提取数据 (CPU Numpy)
            raw_map = depth_tensor[0].detach().cpu().numpy() 
            if raw_map.ndim == 3: raw_map = raw_map.squeeze(0)

            # 3. 标记无效值 (0.0) -> 后面涂绿
            mask_invalid = (raw_map < 0.001)
            
            # 4. 归一化并转伪彩色
            norm_data = np.clip(raw_map / max_r, 0, 1) * 255
            depth_uint8 = norm_data.astype(np.uint8)
            colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
            
            # 5. 将无效值涂成鲜绿色 (Green)
            colored[mask_invalid] = [0, 255, 0] 
            
            # 6. 放大显示 (方便人眼观察)
            display_img = cv2.resize(colored, (400, 300), interpolation=cv2.INTER_NEAREST)
            
            # 7. 打印 Min/Max
            min_val = np.min(raw_map)
            max_val = np.max(raw_map)
            cv2.putText(display_img, f"Range: {min_val:.2f} ~ {max_val:.2f} m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Depth Camera", display_img)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Vis Error: {e}")
        
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
            
        # 2. [关键] 直接从 Tensor 获取真实形状，不依赖可能搞反的配置
        # height_samples 的形状通常是 [num_rows, num_cols]
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
        """
        hanging_mask: (num_envs, num_feet, 4) - 哪个角悬空了
        contact_mask: (num_envs, num_feet, 4) - 是否处于触地状态
        """
        with torch.no_grad():
            # 1. 计算全贴合率 (Full Plantar Contact Rate)
            # 只有当这只脚在接触地面，且 4 个角都没有悬空时，才是 True
            # hanging_mask.any(dim=-1) 表示只要有一个角悬空就是 True
            is_full_contact = contact_mask[:, :, 0] & (~hanging_mask.any(dim=-1))
            
            # 将结果存入 extras，方便后续在 Tensorboard 或日志中查看
            # 我们只统计当前所有正在触地的脚中，有多少是全贴合的
            total_contacts = torch.sum(contact_mask[:, :, 0]).float()
            if total_contacts > 0:
                full_contact_rate = torch.sum(is_full_contact).float() / total_contacts
                self.extras["metrics/full_contact_rate"] = full_contact_rate
    
    def reset_idx(self, env_ids):
        """ 重置环境时的钩子函数 """
        # 如果是在测试模式（play.py），且日志积累到一定量，则保存
        # 这里的 self.cfg.env.test 取决于你运行 play.py 时是否传入了相应参数
        if len(self.foothold_log) > 100000: 
            self.save_foothold_data()
        
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
        
        # 路径：logs/g1/data/
        # log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.cfg.runner.experiment_name, 'data')
        # 直接使用机器人资产名称，通常就是 'g1'
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

    def _reward_foot_flatness(self):
        """
        [新增奖励] 惩罚脚掌与地面不平行的姿态
        鼓励脚掌的 Z 轴（向上）与世界坐标系的 Z 轴对齐
        """
        # 1. 获取脚的世界坐标系旋转 (Quat: x, y, z, w)
        feet_rot = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        
        # 2. 计算脚的向上向量 (Local [0, 0, 1] 变换到世界坐标)
        # 也可以直接提取旋转矩阵的第三列
        from isaacgym.torch_utils import quat_apply
        # 定义局部向上向量
        up_local = torch.tensor([0, 0, 1.0], device=self.device).repeat(self.num_envs, self.feet_num, 1)
        # 变换到世界坐标
        up_world = quat_apply(feet_rot.view(-1, 4), up_local.view(-1, 3)).view(self.num_envs, self.feet_num, 3)
        
        # 3. 计算与世界垂直方向 [0, 0, 1] 的偏差
        # 如果平贴地面，up_world 应为 [0, 0, 1]，偏差为 0
        # 我们只惩罚垂直分量的损失 (1.0 - up_world_z)
        deviation = 1.0 - up_world[..., 2]
        
        # 4. 只有在支撑相（Stance）时才强制要求平整，摆动相（Swing）允许勾脚尖
        # 使用你已有的 leg_phase (0~0.55 为支撑相)
        stance_mask = (self.leg_phase < 0.55).float()
        
        return torch.sum(torch.square(deviation) * stance_mask, dim=1)

    def _reward_foot_support_rect(self):
        """
        [防悬空奖励 - 偏移修正版]
        基于踝关节位置（靠近脚后跟）计算真实的脚掌四个角坐标。
        参数：足长0.20, 足宽0.06, 踝关节距后跟0.03
        """
        # --- 1. 定义几何参数 (局部坐标系) ---
        # 踝关节是原点 (0,0)
        # 前方距离 = 总长 - 后跟偏移 = 0.20 - 0.03 = 0.17
        dist_front = 0.12
        # 后方距离 = -0.03
        dist_back  = -0.05
        # 左右距离 = 宽度 / 2 = 0.06 / 2 = 0.03
        dist_side  = 0.03
        
        edge_threshold = 0.01  # 允许的高度误差(4cm)

        # 定义四个角相对于踝关节的局部坐标 [x, y, z]
        # 顺序: [前左, 前右, 后左, 后右]
        corners_local_list = [
            [dist_front,  dist_side, 0.0], # Front Left (+0.17, +0.03)
            [dist_front, -dist_side, 0.0], # Front Right (+0.17, -0.03)
            [dist_back,   dist_side, 0.0], # Back Left   (-0.03, +0.03)
            [dist_back,  -dist_side, 0.0]  # Back Right  (-0.03, -0.03)
        ]
        
        # 转换为 Tensor
        corners_local = torch.tensor(corners_local_list, device=self.device, dtype=torch.float32)

        # --- 2. 变换到世界坐标系 ---
        feet_pos = self.feet_pos  # (num_envs, num_feet, 3)
        # 获取脚的旋转四元数
        feet_rot = self.rigid_body_states_view[:, self.feet_indices, 3:7]

        # 扩展维度以便广播计算
        # corners_local: (4, 3) -> (num_envs, num_feet, 4, 3)
        num_corners = 4
        
        # (num_envs, num_feet, 1, 3)
        feet_pos_exp = feet_pos.unsqueeze(2) 
        
        # (num_envs, num_feet, 1, 4) -> 重复4次适配4个角
        feet_rot_exp = feet_rot.unsqueeze(2).repeat(1, 1, num_corners, 1)
        
        # (num_envs, num_feet, 4, 3)
        corners_exp = corners_local.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.feet_num, 1, 1)
        
        # 执行旋转 + 平移
        # Flatten后计算再Reshape回来
        corners_world = quat_apply(feet_rot_exp.view(-1, 4), corners_exp.view(-1, 3)).view(self.num_envs, self.feet_num, 4, 3)
        corners_world += feet_pos_exp # 加上踝关节的世界坐标

        # --- 3. 地形高度查询 ---
        # 提取 xy 坐标进行查询
        x_query = corners_world[..., 0].view(-1)
        y_query = corners_world[..., 1].view(-1)
        
        # 查表得到地形高度 (N*F*4)
        terrain_heights = self._get_terrain_heights(None, x_query, y_query).view(self.num_envs, self.feet_num, 4)
        
        # 角的当前真实高度
        corners_z = corners_world[..., 2]
        
        # --- 4. 计算悬空惩罚 ---
        # 计算 高度差 (脚底 - 地形)
        height_diff = corners_z - terrain_heights
        
        # 判定悬空: 高度差 > 阈值 (比如脚在空中4cm以上，且下方地形很深)
        # 注意：这里我们只关心脚处于"Stance"状态时的悬空
        hanging_mask = height_diff > edge_threshold
        
        # 获取接触状态 (Contact Force > 1N)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # 扩展 mask 维度: (num_envs, num_feet, 4)
        contact_mask = contact.unsqueeze(2).repeat(1, 1, 4)
        
        # 核心逻辑：如果脚掌显示接触了地面(Sensor显示受力)，但某个角下方是空的，给予惩罚
        # 这意味着机器人踩在台阶边缘，极其容易滑落
        penalize = torch.sum(torch.square(height_diff) * hanging_mask.float() * contact_mask.float(), dim=(1, 2))
        self._update_contact_metrics(hanging_mask, contact_mask)
        return penalize

    def _reward_stair_clearance(self):
        """
        [正向惩罚版] 表现越差，数值越大 (0 ~ 1)
        逻辑：脚尖低于台阶时输出正值。
        权重设置：负数 (例如 -0.5)
        """
        # 1. 识别摆动腿
        is_swing = self.leg_phase > 0.55
        
        # 2. 预测前方地形
        lin_vel = self.base_lin_vel[:, :2]
        vel_norm = torch.norm(lin_vel, dim=1, keepdim=True)
        heading = lin_vel / (vel_norm + 1e-5)
        
        lookahead_dist = 0.05 
        feet_pos = self.feet_pos
        sample_x = feet_pos[..., 0] + heading[:, 0].unsqueeze(1) * lookahead_dist
        sample_y = feet_pos[..., 1] + heading[:, 1].unsqueeze(1) * lookahead_dist
        
        terrain_h = self._get_terrain_heights(None, sample_x.view(-1), sample_y.view(-1))
        terrain_h = terrain_h.view(self.num_envs, self.feet_num)
        
        # 3. 计算高度误差
        target_h = terrain_h + 0.02 # 安全阈值
        current_h = feet_pos[..., 2]
        error = torch.clip(target_h - current_h, min=0.0)
        
        # 4. 指数惩罚映射: f(x) = 1 - exp(-x / sigma)
        # error=0 时, penalty=0; error很大时, penalty趋近于 1
        sigma = 0.02
        penalty = 1.0 - torch.exp(-error / sigma)
        
        # 5. 掩码过滤
        moving_mask = (vel_norm > 0.1).float()
        res = torch.sum(penalty * is_swing.float(), dim=1)
        
        return res * moving_mask.squeeze(1)
    def _reward_feet_stumble(self):
        """
        [标准版 - 强力防绊倒]
        如果脚受到的 水平撞击力(XY) 远大于 垂直支撑力(Z)，说明踢到了台阶立面。
        """
        # 1. 获取脚部受力 (num_envs, num_feet, 3)
        forces = self.contact_forces[:, self.feet_indices, :]
        
        # 2. 计算水平力和垂直力
        force_xy = torch.norm(forces[:, :, :2], dim=2)
        force_z = torch.abs(forces[:, :, 2])
        
        # 3. 判定绊倒 (Stumble)
        # 条件: 水平力 > 10N 且 水平力 > 垂直力 * 4
        # 这种大力撞击通常发生在踢到台阶垂直面时
        is_stumble = (force_xy > 10.0) & (force_xy > 4.0 * force_z)
        
        # 4. 惩罚
        return torch.sum(is_stumble.float(), dim=1)
    # def _reward_pelvis_height(self):
    #     # 1. 获取基座高度 (num_envs,)
    #     base_height = self.root_states[:, 2]
        
    #     # 2. 获取脚底高度 (num_envs, num_feet)
    #     feet_heights = self.feet_pos[:, :, 2]
    #     print(f"所有脚底高度：{feet_heights}")
    #     # 3. [核心修改] 按照你的逻辑：取脚底坐标的最小值
    #     # 无论哪只脚更低（不管是支撑还是摆动），都以它作为地面参考点
    #     # torch.min 返回 (values, indices)，我们取 [0] 即数值部分
    #     target_ground_height = torch.min(feet_heights, dim=1)[0]
    #     print(f"基座高度：{base_height}")
    #     print(f"脚底高度：{target_ground_height}")
    #     # 4. 计算相对高度 (当前机器人身体伸展的垂直距离)
    #     dist = base_height - target_ground_height
    #     print(f"差值：{dist}")
    #     # 5. 计算误差平方
    #     target_height = 0.75
    #     reward_error = torch.square(dist - target_height)
    #     print(f"误差：{dist - target_height}")
    #     print(f"误差平方：{reward_error}")
    #     # 6. 此时建议取消“腾空置零”的逻辑
    #     # 因为既然以最低脚为准，即便腾空，最低的那只脚也能提供一个有效的参考
    #     return reward_error

    def _reward_pelvis_height(self):
        base_height = self.root_states[:, 2]
        
        # 1. 获取基座下方的地形高度作为“保底参考”
        # 这样即使双脚离地，参考点也是脚下的楼梯，而不是世界原点 0
        terrain_h = self._get_terrain_heights(None, self.root_states[:, 0], self.root_states[:, 1])
        
        # 2. 计算脚踝的加权高度
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        weights = torch.clip(contact_forces / 100.0, 0.0, 1.0)
        sum_weights = torch.sum(weights, dim=1)
        
        # 3. 关键修正：如果没踩地，使用地形高度；如果踩地了，使用脚踝加权高度
        # 这能保证参考基准的连续性
        foot_h_val = torch.sum(self.feet_pos[:, :, 2] * weights, dim=1) / torch.clamp(sum_weights, min=1e-3)
        target_ground_h = torch.where(sum_weights > 0.1, foot_h_val, terrain_h)
        
        # 4. 计算偏差
        dist = base_height - target_ground_h
        target_height = 0.70
        
        # 5. 对于单层网络，强烈建议用 exp 限幅，防止 square 产生过大梯度
        # 这样即便机器人瞬间飞起，惩罚也不会超过权重上限
        error = torch.square(dist - target_height)
        return 1.0 - torch.exp(-error / 0.1)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
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
        # 计算当前关节角与默认关节角的差值
        # 这样机器人保持在 -0.1 (默认姿态) 时，奖励最高，不再“纠结”
        error = self.dof_pos - self.default_dof_pos
        
        # 假设 1,2 是左腿 Roll/Pitch; 7,8 是右腿 Roll/Pitch
        # 请确保索引与你的 DOF 顺序一致
        return torch.sum(torch.square(error[:, [1, 2, 7, 8]]), dim=1)
