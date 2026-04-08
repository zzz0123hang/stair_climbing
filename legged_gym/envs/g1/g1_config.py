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
            # 视觉“看到台阶线索”判定（用于 vision_stair_drive 的增强门控）
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
        # 平地重采样时给少量“零命令”样本，训练无指令静止（避免占比过大影响主任务）
        flat_idle_prob = 0.12
        # 非零样本里前向牵引比例
        flat_forward_prob = 0.78
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
        # Play 手动控制模式：开启后，环境不再覆盖外部下发的速度/角速度指令
        manual_cmd_override = False
        # num_observations = 47 + 160
        # num_privileged_obs = 50 + 160
        num_observations = 790#本体感受(50)*3帧历史 + 视觉特征(640) = 790
        num_privileged_obs = 980#学生(790) + 局部高度采样(187) + 物理参数(3) = 980
        

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
        late_stage_start_level = 5.0
        late_stage_progress_pow = 1.6
        # 升级距离阈值：保持严格标准，要求接近完成一段完整上坡推进
        curriculum_move_up_distance = 3.5
        # 升级还需满足 final_distance > move_up_distance * ratio（抗推搡误升）
        curriculum_move_up_final_ratio = 0.60
        # 降级距离阈值（终点位移）
        curriculum_move_down_distance = 1.0
        # 降级还需满足 max_distance < move_up_distance * ratio（防“有推进却回摆”误降）
        curriculum_move_down_max_ratio = 0.55
        # 升级还需满足回合内“楼梯暴露占比”达到阈值，减少“平地走远就升级”
        curriculum_move_up_stair_conf = 0.10
        # 峰值兜底阈值
        curriculum_move_up_stair_conf_peak = 0.40
        # 最少导航步数，避免偶发短时暴露触发误升
        curriculum_move_up_nav_steps_min = 28
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
        max_push_vel_xy = 0.5
        # 推搡课程化：前期几乎无推搡，后期逐步拉起
        push_curriculum = True
        push_start_progress = 0.45
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
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # --- 基础运动奖励 ---
            tracking_lin_vel = 1.2
            # 轻保底，避免“活着就有高分”掩盖主任务
            alive = 0.08
            # 关闭：与 stair_clearance 目标高度引导重叠，容易形成抬腿刷分
            feet_air_time = 0.0
            # [正向] 在非楼梯区鼓励“朝向台阶并持续前进”（实际强度由课程进一步调度）
            approach_stairs = 0.4
            # ================= 核心惩罚 (全部必须为负数!) =================
            ang_vel_xy = -0.05          # [负向] 惩罚身体摇晃
            dof_acc = -1.5e-7           # [负向] 惩罚关节加速度过大 (省电、平滑)
            # 抑制“无指令漂移”和“有指令怠工”
            stand_still = -0.12
            # ================= 楼梯专项惩罚 (必须为负数!) =================
            # 4. 惩罚摆动腿离地高度偏离目标值 (0.08m)
            # feet_swing_height = -1.5             
            # 7. 惩罚髋关节劈叉或奇怪的姿势
            # 小权重保留：约束髋部姿态漂移，但不过度压制上台阶代偿
            hip_pos = -0.2
            # 8. 惩罚脚落地后还在滑动 (打滑惩罚)
            # 关闭：与 rect/clearance/contact 等形成重复约束，易抑制落脚后微调
            contact_no_vel = 0.0

            dof_pos_limits = -3.0
            #######################
            orientation = -0.3#-0.5          # [负向] 惩罚基座倾斜
            # 1. 惩罚脚掌与地面不平行 (专治脚尖绷直、蝎子摆尾)
            # 关闭：与 foot_support_rect 目标重叠，先释放探索空间
            foot_flatness = 0.0
            
            pelvis_height = 0.10

            # 视觉对齐惩罚：提高权重，抑制“看见台阶仍绕圈”
            vision_stair_drive = -0.45


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
            foot_support_rect = -0.35   #-2.0
            stair_clearance = 0.24
            feet_stumble = -1.0
            stair_alternation = 0.45
            
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
            # vision_stair_drive = 1.5   
            
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
        learning_rate = 5e-4#5e-4      # 从 1e-3 降到 5e-4
        num_mini_batches = 4#8  # 增加分块以稳定显存和梯度
        entropy_coef = 0.01#0.02#0.01       # 从 0.01 提到 0.02，防止过早收敛到小碎步策略
        # entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTS"
        max_iterations = 5000
        #num_steps_per_env没有编辑，默认为24
        num_steps_per_env = 64
        run_name = "TS_S_Climbing"
        experiment_name = 'g1'
