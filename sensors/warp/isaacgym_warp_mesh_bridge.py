# sensors/warp/isaacgym_warp_mesh_bridge.py

import warp as wp
import numpy as np

class IsaacGymWarpMeshBridge:
    def __init__(self, gym, sim, vertices=None, triangles=None, device='cuda'):
        self.gym = gym
        self.sim = sim
        self.device = device
        
        # 存储 Mesh 的 ID
        self.mesh_ids = []
        
        # 【关键修改】直接使用传入的顶点和三角形数据，而不是去 Gym 里调用不存在的函数
        if vertices is not None and triangles is not None:
            self._create_mesh_from_data(vertices, triangles)
        else:
            print("[Warning] IsaacGymWarpMeshBridge: No terrain data provided. Warp camera will render empty space.")

    def _create_mesh_from_data(self, vertices, triangles):
        """
        使用从外部传入的 Numpy 数据创建 Warp Mesh
        """
        try:
            # 1. 确保数据类型正确 (Warp非常挑剔数据类型)
            # 顶点必须是 float32
            vertices = np.array(vertices, dtype=np.float32)
            # 索引必须是 int32
            triangles = np.array(triangles, dtype=np.int32)
            
            # 2. 将数据上传到 GPU (Warp Memory)
            # Warp Mesh 需要 points (vec3) 和 indices (int)
            wp_points = wp.from_numpy(vertices, dtype=wp.vec3, device=self.device)
            
            # Warp 期待 indices 是展平的一维数组 (M*3,)
            # 无论输入是 (M,3) 还是 (M*3,)，flatten() 都能处理
            wp_indices = wp.from_numpy(triangles.flatten(), dtype=wp.int32, device=self.device)
            
            # 3. 创建 Warp Mesh 对象
            self.mesh = wp.Mesh(
                points=wp_points,
                indices=wp_indices,
                velocities=None
            )
            
            # 4. 存入 ID 列表 (WarpStereoCam 需要这个列表)
            self.mesh_ids.append(self.mesh.id)
            
            print(f"[IsaacGymWarpMeshBridge] Successfully created terrain mesh with {len(vertices)} vertices.")
            
        except Exception as e:
            print(f"[Error] Failed to create Warp mesh: {e}")

    def get_mesh_ids_list(self, num_envs=1):
        """
        [优化版] 返回 WarpStereoCam 初始化所需的 mesh_ids_array
        参数:
            num_envs: 需要广播的环境数量。如果不传，默认为1。
        """
        if not self.mesh_ids:
            return wp.array([], dtype=wp.uint64, device=self.device)
            
        # 1. 获取基础 ID 列表 (通常只有一个)
        ids_np = np.array(self.mesh_ids, dtype=np.uint64)
        
        # 2. 智能广播逻辑
        # 如果只有一个地形 ID，但有多个环境，我们需要把这个 ID 复制 num_envs 份
        if len(ids_np) == 1 and num_envs > 1:
            # 使用 Warp 高效填充
            unique_id = ids_np[0]
            return wp.full(shape=(num_envs,), value=unique_id, dtype=wp.uint64, device=self.device)
        
        # 如果 ID 数量已经匹配 (比如未来做多地形)，或者是单环境，直接返回
        return wp.from_numpy(ids_np, dtype=wp.uint64, device=self.device)

    def update(self):
        # 如果地形是静态的，这里不需要做任何事
        pass