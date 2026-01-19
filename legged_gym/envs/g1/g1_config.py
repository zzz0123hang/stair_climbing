from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    # ================= [新增] 视觉传感器配置 =================
    class sensor:
        class stereo_cam:
            # [关键修复] 补回遗漏的传感器数量参数
            num_sensors = 1         
            
            # --- D435i 参数 ---
            width = 32#16             # 12x8 分辨率 (1.5:1 比例)
            height = 20#10            
            horizontal_fov_deg = 87
            # horizontal_fov_deg = 85.6 
            max_range = 3.0       
            calculate_depth = True  
            baseline = 0.05   
            visual_feature_dim = 640      
            segmentation_camera = False 
            return_pointcloud = False   
            pointcloud_in_world_frame = False
            # --- 挂载参数 ---
            enable = True
            parent_link = "pelvis"
            local_pos = [0.05366, 0.01753, 0.47387] 
            local_rpy = [0.0, 0.58, -0.022]
            # local_rpy = [0.0, 0.58, 0.0]
            # local_rpy = [0.0, 0.830776724, 0.0]
            # local_rpy = [0.0, 0.5+1.57, 0.0]
            
            # --- 噪声与频率 ---
            noise_level = 0.0 
            dropout_prob = 0.0
            update_interval = 1
           
    # =======================================================

    class init_state( LeggedRobotCfg.init_state ):
        # pos = [4.0, 4.0, 0.8]
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        # [修改] 增加视觉观测维度 (+64)
        num_envs = 1024
        num_observations = 47 + 640
        num_privileged_obs = 50 + 640
        num_actions = 12

    # [恢复] 地形配置 (这对视觉训练很重要)
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # curriculum = True
        mesh_type = 'plane'
        curriculum = False
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]

        terrain_proportions = [0.4, 0.2, 0.2, 0.2, 0.0]
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
        num_rows= 10 
        num_cols = 20 
        # num_rows= 5 
        # num_cols = 5 
        
        slope_treshold = 0.75

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis","hip"]
        self_collisions = 0 
        flip_visual_attachments = False
        # collapse_fixed_joints = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5#1.0#这两个奖励大了，机器人就会岔开腿
            tracking_ang_vel = 1.0#0.8#这两个奖励大了，机器人就会岔开腿
            lin_vel_z = -1.0#-2.0#这个惩罚小了，机器人就会跳步
            ang_vel_xy = -0.02#-0.05
            orientation = -1.0#机器人在水平面的投影，所以机器人越直，这个惩罚越小(解决屈膝问题)
            dof_acc = -1.0e-7#-2.5e-7
            dof_vel = -5e-4#-1e-3
            feet_air_time = 0.5#0.2
            collision = 0.0
            action_rate = -0.02#-0.02
            dof_pos_limits = -2.0#-5.0
            alive = 0.8#0.15
            hip_pos = -1.0#惩罚机器人的hip关节角，角度越大惩罚越大
            contact_no_vel = -0.2
            contact = 0.5#0.8#0.18#小了的话，这个小机器人的左右腿就会不对称
            #############
            pelvis_height = -1.0#基座相对高度正向奖励
            foot_flatness = -0.2
            foot_support_rect = -0.5 #-1.0
            stair_clearance = -0.5#-1.0
            feet_stumble = -0.5#-1.0
            #############################################################
            # tracking_lin_vel = 1.5#1.2#这两个奖励大了，机器人就会岔开腿
            # tracking_ang_vel = 1.0#这两个奖励大了，机器人就会岔开腿
            # lin_vel_z = -0.5#这个惩罚小了，机器人就会跳步
            # ang_vel_xy = -0.1#-0.05
            # orientation = -3.0#-1.0#机器人在水平面的投影，所以机器人越直，这个惩罚越小
            # dof_acc = -2.5e-7
            # dof_vel = -5e-4#-1e-3
            # feet_air_time = 0.2
            # collision = 0.0
            # action_rate = -0.05#-0.02
            # dof_pos_limits = -5.0
            # alive = 0.3#0.15
            # hip_pos = -0.1#-1.0#惩罚机器人的hip关节角，角度越大惩罚越大
            # contact_no_vel = -0.2
            # contact = 0.8#0.18#这个小机器人的左右腿就会不对称
            # #############
            # pelvis_height = -1.5#-1.0
            # foot_support_rect = -2.0 #-1.0
            # stair_clearance = -1.5#-1.0
            # feet_stumble = -0.5#-1.0

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0#0.8
        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' 
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        # rnn_hidden_size = 64
        rnn_hidden_size = 256
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 5e-4      # 从 1e-3 降到 5e-4
        num_mini_batches = 4#8  # 增加分块以稳定显存和梯度
        entropy_coef = 0.01#0.02#0.01       # 从 0.01 提到 0.02，防止过早收敛到小碎步策略
        # entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 5000
        #num_steps_per_env没有编辑，默认为24
        num_steps_per_env = 32
        run_name = ''
        experiment_name = 'g1'