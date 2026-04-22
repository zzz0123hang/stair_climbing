from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    # ================= [新增] 视觉传感器配置 =================
    class sensor:
        class stereo_cam:
            # [关键修复] 补回遗漏的传感器数量参数
            num_sensors = 1         
            # --- D435i 参数 ---
            # width = 16             # 12x8 分辨率 (1.5:1 比例)
            # height = 10
            width = 32             # 12x8 分辨率 (1.5:1 比例)
            height = 20           
            horizontal_fov_deg = 87
            # horizontal_fov_deg = 85.6 
            max_range = 3.0       
            calculate_depth = True  
            baseline = 0.05  
            # visual_feature_dim = 160 
            visual_feature_dim = 640      
            segmentation_camera = False 
            return_pointcloud = False   
            pointcloud_in_world_frame = False
            # --- 挂载参数 ---
            enable = True
            parent_link = "pelvis"
            local_pos = [0.05366, 0.01753, 0.47387] 
            # local_rpy = [0.0, 0.58, -0.022]
            local_rpy = [0.0, 0.58, 0.0]
            # local_rpy = [0.0, 0.830776724, 0.0]
            # local_rpy = [0.0, 0.5+1.57, 0.0]
            
            # --- 噪声与频率 ---
            noise_level = 0.0 
            dropout_prob = 0.0
            update_interval = 1
            # 开阔平地时视觉控制权的最大释放比例（越小越“听视觉纠偏”）
            max_open_release = 0.45
            # 课程末期视觉纠偏保留比例（1.0=不下放；0.65=保留65%视觉接管）
            visual_handover_end_gain = 0.65
            # 对齐奖励全程底座门控，避免远离楼梯时完全无约束
            align_global_floor = 0.12
            # 视觉“看到台阶线索”判定（用于 stair_alignment 的增强门控）
            vision_seen_depth_thresh = 1.25
            vision_seen_depth_sigma = 0.18
            vision_seen_edge_thresh = 0.10
            vision_seen_edge_sigma = 0.03
           
    # =======================================================
    class normalization( LeggedRobotCfg.normalization ):
            class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
                # 这个 5.0 是业界标准值，用于将高度差缩放到网络容易处理的范围
                height_measure = 5.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 3. # time before command are changed[s]
        heading_command = False#heading_command = True # if true: compute ang vel command from heading error
        # 平地随机角速度范围（_resample_commands 中使用）
        flat_yaw_range = 0.1
        # 平地零命令样本占比（受全局 no_cmd governor 约束）
        flat_idle_prob = 0.14
        # 非零样本里前向牵引比例
        flat_forward_prob = 0.78
        # 楼梯牵引最小前进速度（过高会抑制交替稳定并提高摔倒率）
        stair_min_forward = 0.30
        # 楼梯区保留少量零指令样本：静止能力不断档，但不压过上楼主任务
        stair_idle_prob = 0.03
        stair_idle_prob_early = 0.05
        stair_idle_prob_late = 0.02
        # 全局 no_cmd 比例目标带：主任务优先，同时保留静止能力
        target_no_cmd_rate = 0.12
        target_no_cmd_rate_low = 0.08
        target_no_cmd_rate_high = 0.16
        # 楼梯牵引样本里的 no_cmd 硬上限（剩余 no_cmd 配额优先分给 flat）
        stair_idle_hard_cap_ratio = 0.05
        # 默认零指令连续片段锁存时长（秒）：用于平地静止训练
        no_cmd_hold_min_s = 2.0
        no_cmd_hold_max_s = 3.2
        # 楼梯活跃样本上的零指令短锁存：保留静止能力，同时不打断爬楼主任务
        stair_no_cmd_hold_min_s = 0.6
        stair_no_cmd_hold_max_s = 1.2
        # 平地静止样本锁存（与默认一致，显式化便于调参）
        flat_no_cmd_hold_min_s = 2.0
        flat_no_cmd_hold_max_s = 3.2
        # 统一零指令判据阈值（所有 no_cmd 门控共用）
        no_cmd_planar_thr = 0.05
        no_cmd_yaw_thr = 0.05
        class ranges:
            lin_vel_x = [-0.5, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            # heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        # pos = [12.0, 4.0, 0.8]
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : 0.0, 
        #    'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,          
           'right_ankle_pitch_joint': 0.0,                                    
        #    'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        # [修改] 增加视觉观测维度 (+64)
        num_envs = 1024
        num_actions = 12
        # 课程升降级详细打印（默认关闭，避免大规模并行训练被日志拖慢）
        log_curriculum_events = False
        # 课程 gate 细分指标：先关闭，减少训练日志开销
        log_curriculum_gate_details = False
        # 距离明细指标：先关闭，必要时再用于定位问题
        log_distance_detail_metrics = False
        # 回合奖励曲线白名单：保留核心训练信号 + 少量可排障分项
        episode_reward_log_keys = [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "stand_still",
            "stair_clearance",
            "stair_alternation",
            "planner_tracking",
            "foot_support_rect",
            "first_step_commit",
            "orientation",
            "pelvis_height",
            "contact",
            "termination",
            "alive",
            "action_rate",
        ]
        # 实时 extras 白名单：保留课程推进、零指令稳定性、planner 闭环与调度关键信号
        # Metrics/ 走前缀放行：恢复原有课程/地形相关曲线，避免 key 变动时被白名单误杀。
        wandb_extra_prefixes_keep = [
            "Metrics/",
        ]
        wandb_extra_keys_keep = [
            "metrics/full_contact_rate",
            "Diagnostics/sched_planner_miss_boost",
            "Diagnostics/sched_no_cmd_boost",
            "Diagnostics/no_cmd_rate",
            "Diagnostics/no_cmd_hold_rate",
            "Diagnostics/no_cmd_yaw_rate_mean",
            "Diagnostics/no_cmd_instability_ema",
            "Diagnostics/no_cmd_single_support_rate",
            "Diagnostics/no_cmd_double_flight_rate",
            "Diagnostics/planner_xy_err_ema",
            "Diagnostics/planner_touch_count_step",
            "Diagnostics/planner_hit8_ema",
            "Diagnostics/planner_hit4_ema",
            "Diagnostics/reset_rate_step",
            "Diagnostics/reset_no_cmd_share_step",
            "Diagnostics/reset_move_cmd_share_step",
            "Diagnostics/reset_episode_len_no_cmd_ema",
            "Diagnostics/reset_episode_len_move_cmd_ema",
            "Diagnostics/reset_reason_contact_share_cum",
            "Diagnostics/reset_reason_pose_share_cum",
            "Diagnostics/reset_reason_timeout_share_cum",
        ]
        # Play 手动控制模式：开启后，环境不再覆盖外部下发的速度/角速度指令
        manual_cmd_override = False
        # Play/Test 模式下，reset 后是否允许自动随机重采样命令
        allow_test_resample = False
        # 姿态终止阈值（弧度）：与上游逻辑对齐
        terminate_pitch = 1.0
        terminate_roll = 0.8
        terminate_pose_consecutive_steps = 1
        terminate_contact_force = 2.0
        terminate_contact_consecutive_steps = 2
        # num_observations = 47 + 160
        # num_privileged_obs = 50 + 160
        num_observations = 798#本体感受历史(150) + planner对齐特征(8) + 视觉特征(640)
        num_privileged_obs = 988#学生(798) + 局部高度采样(187) + 物理参数(3) = 988
        

    # [恢复] 地形配置 (这对视觉训练很重要)
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        # mesh_type = 'plane'
        # curriculum = True 
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]

        # terrain_proportions = [0.1, 0.1, 0.5, 0.3, 0.0]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.2, 0.2, 0.3, 0.3, 0.0]
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        # terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        # measure_heights = True 开启以便与视觉信息互补或对比
        measure_heights = True 
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False 
        terrain_kwargs = None 
        max_init_terrain_level = 1 
        terrain_length = 8.
        terrain_width = 8.
        # 后期课程从更高等级才开始生效，避免“学会走之前就进严格评分”
        late_stage_start_level = 7.0
        late_stage_progress_pow = 2.2
        # 升级距离阈值：保持严格标准，要求接近完成一段完整上坡推进
        curriculum_move_up_distance = 4.5
        # 升级还需满足 final_distance > move_up_distance * ratio（抗推搡误升）
        curriculum_move_up_final_ratio = 0.76
        # 降级距离阈值（终点位移）
        curriculum_move_down_distance = 1.0
        # 降级还需满足 max_distance < move_up_distance * ratio（防“有推进却回摆”误降）
        curriculum_move_down_max_ratio = 0.55
        # 当前为纯楼梯金字塔地形，先关闭 stair_conf 升级门控，简化课程判据
        curriculum_move_up_stair_conf = 0.0
        # 峰值兜底阈值
        curriculum_move_up_stair_conf_peak = 0.52
        # 最少导航步数，避免偶发短时暴露触发误升
        curriculum_move_up_nav_steps_min = 56
        num_rows= 10 
        num_cols = 20 
        # num_rows= 5 
        # num_cols = 5 
        
        slope_treshold = 0.75

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        # friction_range = [0.1, 1.25]
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 8
        # 重置后先给若干稳定时间，避免“无指令静止样本”被即时推搡污染
        push_warmup_s = 1.0
        max_push_vel_xy = 0.5
        # 推搡课程化：前期几乎无推搡，后期逐步拉起
        push_curriculum = True
        push_start_progress = 0.85
        push_min_scale = 0.0

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'hip_yaw': 80,    # 略微降低
        #              'hip_roll': 80,
        #              'hip_pitch': 80,
        #              'knee': 100,      # 从 150 降低到 100，允许膝盖有弹性地弯曲
        #              'ankle': 60,      # 从 40 降低到 30
        #              }
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 3,        # 相应降低阻尼
        #              'ankle': 2,
        #              }
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 100,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 4,
                    #  'ankle': 2,
                     }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.3#0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "thigh"]
        ####################################################################################
        terminate_after_contacts_on = ["pelvis"]#"hip"
        self_collisions = 0 
        flip_visual_attachments = False
        # collapse_fixed_joints = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.72
        only_positive_rewards = False
        # 探索样本任务降权系数（1.0=不降权，0.0=完全屏蔽）
        explore_task_weight = 0.30
        # 全贴合判定阈值：脚角点离地超过该高度视为悬空（米）
        full_contact_edge_threshold = 0.01
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # --- 基础运动奖励 ---
            tracking_lin_vel = 1.2
            # 显式惩罚摔倒终止，避免“猛冲几步后前扑”成为局部最优
            termination = -8.0
            # 轻保底，避免“活着就有高分”掩盖主任务
            alive = 0.03
            # 关闭：与 stair_clearance 目标高度引导重叠，容易形成抬腿刷分
            feet_air_time = 0.0
            # [正向] 在非楼梯区鼓励“朝向台阶并持续前进”（实际强度由课程进一步调度）
            approach_stairs = 0.4
            # ================= 核心惩罚 (全部必须为负数!) =================
            ang_vel_xy = -0.05          # [负向] 惩罚身体摇晃
            dof_acc = -1.5e-7           # [负向] 惩罚关节加速度过大 (省电、平滑)
            # 抑制“无指令漂移”和“有指令怠工”
            stand_still = -0.12
            # ================= 楼梯专项（含奖励与惩罚） =================
            # 4. 惩罚摆动腿离地高度偏离目标值 (0.08m)
            # feet_swing_height = -1.5             
            # 7. 惩罚髋关节劈叉或奇怪的姿势
            # 小权重保留：约束髋部姿态漂移，但不过度压制上台阶代偿
            hip_pos = -0.2
            # 8. 惩罚脚落地后还在滑动 (打滑惩罚)
            # 小权重保留：用于稳住零指令站立，不与 rect/clearance 主约束抢权重
            contact_no_vel = -0.04

            dof_pos_limits = -3.0
            #######################
            orientation = -0.3#-0.5          # [负向] 惩罚基座倾斜
            # 1. 惩罚脚掌与地面不平行 (专治脚尖绷直、蝎子摆尾)
            # 小权重保留：强化“全脚掌贴合”，特别是静止与落足末段
            foot_flatness = -0.10
            
            pelvis_height = 0.10

            # 视觉对齐惩罚：提高权重，抑制“看见台阶仍绕圈”
            stair_alignment = -0.45


            # tracking_ang_vel = 2.0
            # contact = 0.5
            # first_step_commit = 0.5
            # lin_vel_z = -0.35         # [负向] 惩罚身体在垂直方向的剧烈起伏 (跳步)
            # action_rate = -0.015      # [负向] 惩罚高频抖动
            # foot_support_rect = -0.2  #-2.0
            # stair_clearance = 0.3
            # feet_stumble = -0.8
            # stair_alternation = 0.3
            tracking_ang_vel = 1.8
            contact = 0.45
            first_step_commit = 0.4
            lin_vel_z = -0.45           # [负向] 抑制双脚同跳/过大起伏
            action_rate = -0.02         # [负向] 进一步抑制抖动和碎步
            # 台阶内安全落脚奖励：函数内部已含负惩罚分支，因此权重为正
            foot_support_rect = 0.35
            stair_clearance = 0.24
            feet_stumble = -1.0
            stair_alternation = 0.45
            # Planner 跟踪（XY-only）：约束实际落足贴合规划落点；Z 由 clearance 负责
            planner_tracking = 0.20
            
            # 权重给到 1.0，强力纠正它侧着身子蹭台阶的坏习惯
            # stair_alignment = -0.5
            #######
            # orientation = -0.5
            # foot_flatness = -1.5
            # foot_support_rect = -2.0
            # stair_clearance = -1.5
            # 5. 强力防绊倒惩罚 (发生水平强冲击时扣分)
            # feet_stumble = -2.0
            # pelvis_height = -1.0

            # --- 新增：导航引导 ( carrot 诱饵 ) ---
            # 权重给到 1.5，让它与前向速度奖励(2.0)叠加，产生巨大的“吸力”
            # stair_alignment = 1.5   
            
            # 权重给到 1.0，强力纠正它侧着身子蹭台阶的坏习惯
            # stair_alignment = 1.0

            #############################################################
            # tracking_lin_vel = 1.2#1.0#这两个奖励大了，机器人就会岔开腿
            # tracking_ang_vel = 0.5#0.8#这两个奖励大了，机器人就会岔开腿
            # lin_vel_z = -1.0#-2.0#这个惩罚小了，机器人就会跳步
            # ang_vel_xy = -0.05#-0.05
            # orientation = -0.5#机器人在水平面的投影，所以机器人越直，这个惩罚越小(解决屈膝问题)
            # dof_acc = -2.5e-7#-2.5e-7
            # dof_vel = -5e-4#-1e-3
            # feet_air_time = 1.0#0.2
            # collision = 0.0
            # action_rate = -0.2#-0.02
            # dof_pos_limits = -5.0#-5.0
            # alive = 0.15#0.15
            # hip_pos = -0.1#惩罚机器人的hip关节角，角度越大惩罚越大
            # contact_no_vel = -0.2
            # contact = 0.5#0.8#0.18#小了的话，这个小机器人的左右腿就会不对称
            # #############
            # # foot_flatness = -0.2
            # pelvis_height = -1.0#基座相对高度正向奖励
            # foot_support_rect = -0.5 #-1.0
            # stair_clearance = -0.5#-1.0
            # feet_stumble = -0.5#-1.0
            # #############################################################
            # # tracking_lin_vel = 1.5#1.2#这两个奖励大了，机器人就会岔开腿
            # # tracking_ang_vel = 1.0#这两个奖励大了，机器人就会岔开腿
            # # lin_vel_z = -0.5#这个惩罚小了，机器人就会跳步
            # # ang_vel_xy = -0.1#-0.05
            # # orientation = -3.0#-1.0#机器人在水平面的投影，所以机器人越直，这个惩罚越小
            # # dof_acc = -2.5e-7
            # # dof_vel = -5e-4#-1e-3
            # # feet_air_time = 0.2
            # # collision = 0.0
            # # action_rate = -0.05#-0.02
            # # dof_pos_limits = -5.0
            # # alive = 0.3#0.15
            # # hip_pos = -0.1#-1.0#惩罚机器人的hip关节角，角度越大惩罚越大
            # # contact_no_vel = -0.2
            # # contact = 0.8#0.18#这个小机器人的左右腿就会不对称
            # # #############
            # # pelvis_height = -1.5#-1.0
            # # foot_support_rect = -2.0 #-1.0
            # # stair_clearance = -1.5#-1.0
            # # feet_stumble = -0.5#-1.0

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.5#1.0#0.8
        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        # actor_hidden_dims = [512]
        # critic_hidden_dims = [512]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' 
        policy_class_name = "ActorCriticTS"
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 256
        # # rnn_hidden_size = 256
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 3e-4#5e-4      # 从 1e-3 降到 5e-4
        num_mini_batches = 4#8  # 增加分块以稳定显存和梯度
        entropy_coef = 0.01#0.02#0.01       # 从 0.01 提到 0.02，防止过早收敛到小碎步策略
        # entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTS"
        max_iterations = 8000
        #num_steps_per_env没有编辑，默认为24
        num_steps_per_env = 64
        run_name = "TS_S_Climbing"
        experiment_name = 'g1'
