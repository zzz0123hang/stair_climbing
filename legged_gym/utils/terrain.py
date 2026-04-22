import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   

        self._build_stair_edge_lookup_maps()
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)

    @staticmethod
    def _scan_prev_inclusive(edge_mask: np.ndarray, axis: int) -> np.ndarray:
        edge = np.moveaxis(edge_mask, axis, 0)
        length = edge.shape[0]
        out = np.full(edge.shape, -1, dtype=np.int32)
        last = np.full(edge.shape[1], -1, dtype=np.int32)
        for i in range(length):
            last = np.where(edge[i], i, last)
            out[i] = last
        return np.moveaxis(out, 0, axis)

    @staticmethod
    def _scan_next_strict(edge_mask: np.ndarray, axis: int) -> np.ndarray:
        edge = np.moveaxis(edge_mask, axis, 0)
        length = edge.shape[0]
        out = np.full(edge.shape, -1, dtype=np.int32)
        nxt = np.full(edge.shape[1], -1, dtype=np.int32)
        for i in range(length - 1, -1, -1):
            out[i] = nxt
            nxt = np.where(edge[i], i, nxt)
        return np.moveaxis(out, 0, axis)

    def _build_stair_edge_lookup_maps(self):
        """
        预计算“最近台阶边缘索引”查表：
        - x/y 轴
        - 前向符号 + / -
        - back / front 最近边缘
        边缘索引定义在栅格坐标系，运行时可 O(1) 查表换算到世界坐标。
        """
        h = self.height_field_raw.astype(np.int32, copy=False)
        vertical_scale = max(float(self.cfg.vertical_scale), 1e-6)
        edge_h_thr = float(getattr(self.cfg, "edge_height_threshold", 0.03))
        raw_thr = max(1, int(np.ceil(edge_h_thr / vertical_scale)))

        dx = h[1:, :] - h[:-1, :]   # 边界索引 i: 在 x=i 处跨越 (i-1 -> i)
        dy = h[:, 1:] - h[:, :-1]   # 边界索引 j: 在 y=j 处跨越 (j-1 -> j)

        edge_x_pos = np.zeros_like(h, dtype=bool)
        edge_x_neg = np.zeros_like(h, dtype=bool)
        edge_y_pos = np.zeros_like(h, dtype=bool)
        edge_y_neg = np.zeros_like(h, dtype=bool)

        edge_x_pos[1:, :] = dx >= raw_thr
        edge_x_neg[1:, :] = (-dx) >= raw_thr
        edge_y_pos[:, 1:] = dy >= raw_thr
        edge_y_neg[:, 1:] = (-dy) >= raw_thr

        back_x_pos = self._scan_prev_inclusive(edge_x_pos, axis=0)
        front_x_pos = self._scan_next_strict(edge_x_pos, axis=0)
        back_x_neg = self._scan_next_strict(edge_x_neg, axis=0)
        front_x_neg = self._scan_prev_inclusive(edge_x_neg, axis=0)

        back_y_pos = self._scan_prev_inclusive(edge_y_pos, axis=1)
        front_y_pos = self._scan_next_strict(edge_y_pos, axis=1)
        back_y_neg = self._scan_next_strict(edge_y_neg, axis=1)
        front_y_neg = self._scan_prev_inclusive(edge_y_neg, axis=1)

        max_dim = max(h.shape[0], h.shape[1])
        index_dtype = np.int16 if max_dim < np.iinfo(np.int16).max else np.int32
        self.stair_edge_lookup = {
            "back_x_pos": back_x_pos.astype(index_dtype, copy=False),
            "front_x_pos": front_x_pos.astype(index_dtype, copy=False),
            "back_x_neg": back_x_neg.astype(index_dtype, copy=False),
            "front_x_neg": front_x_neg.astype(index_dtype, copy=False),
            "back_y_pos": back_y_pos.astype(index_dtype, copy=False),
            "front_y_pos": front_y_pos.astype(index_dtype, copy=False),
            "back_y_neg": back_y_neg.astype(index_dtype, copy=False),
            "front_y_neg": front_y_neg.astype(index_dtype, copy=False),
        }
        self.stair_edge_raw_threshold = raw_thr
        self.stair_edge_height_threshold = raw_thr * vertical_scale
    
    def _resolve_stair_step_height(self, difficulty, row_idx=None):
        """
        Resolve stair step height.
        Priority:
        1) cfg.stair_height_levels (explicit row-wise height list, meters)
        2) default curriculum mapping: 0.05 + 0.1 * difficulty
        """
        levels = getattr(self.cfg, "stair_height_levels", None)
        if levels is not None and len(levels) > 0:
            levels_np = np.asarray(levels, dtype=np.float32).reshape(-1)
            if row_idx is not None:
                idx = int(np.clip(row_idx, 0, len(levels_np) - 1))
            else:
                idx = int(np.clip(np.round(difficulty * (len(levels_np) - 1)), 0, len(levels_np) - 1))
            return float(levels_np[idx])
        return 0.05 + 0.1 * difficulty

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty, row_idx=i)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, row_idx=i)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty, row_idx=None):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = self._resolve_stair_step_height(difficulty, row_idx=row_idx)
        # step_height = 0.12
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.3, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # Use the exact center point height instead of max of 2m x 2m region
        center_x = terrain.height_field_raw.shape[0] // 2
        center_y = terrain.height_field_raw.shape[1] // 2
        env_origin_z = terrain.height_field_raw[center_x, center_y] * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
