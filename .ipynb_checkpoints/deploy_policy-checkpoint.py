# import packages and module here
import sys
import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
import yaml
from datetime import datetime
import importlib

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()
# =================================================================
# 导入我们封装好的 DINOv2 语义提取器 
# (请确保 semantic_extractor.py 就在当前目录下)
# =================================================================
try:
    from semantic_extractor import SemanticPointExtractor
except ImportError:
    print("⚠️ 未找到 semantic_extractor.py，请确保已创建该文件。")


def encode_obs(observation, model=None):  
    """ Post-Process Observation (包含零点坍缩修复) """
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    # 1. 提取点云
    pcd = observation['pointcloud']
    
    # 2. 【核心修复】：解决 DP3 零点坍缩问题 (Point Starvation)
    # 只有当启用了 DINOv2 时，点云才可能因为掩膜过滤而变得极其稀疏
    if True:
        target_num = 1024  # 必须与你 DP3 config 中的点云数完全一致
        current_num = pcd.shape[0]
        #print(current_num)
        if current_num == 0:
            print("⚠️ 警告：当前帧 DINOv2 未提取到任何目标！")
            # 灾难兜底：避免 Numpy 报错，只能全零填充
            pcd_fixed = np.zeros((target_num, pcd.shape[1]), dtype=np.float32)
            
        elif current_num < target_num:
            # 匮乏分支 -> 【有放回的随机重采样】 (避免补零导致 PointNet 重心偏移)
            # indices = np.random.choice(current_num, target_num, replace=True)
            # pcd_fixed = pcd[indices]
            pad_num = target_num - current_num
            padding = np.zeros((pad_num, pcd.shape[1]), dtype=np.float32)
            pcd_fixed = np.vstack([pcd, padding])
            
        elif current_num > target_num:
            try:
                # 提取前三维 (X, Y, Z) 送去跑 FPS
                pcd_xyz = pcd[:, :3]
                
                # 调用你的 pytorch3d fps 接口
                _, indices_tensor = fps(pcd_xyz, target_num, use_cuda=True)
                
                # 解析返回的 tensor 索引
                indices = indices_tensor.detach().cpu().numpy()[0]
                pcd_fixed = pcd[indices]
                print(pcd_fixed.shape[0])
                
            except Exception as e:
                # 如果 pytorch3d 崩了或者没导入成功，作为兜底才使用随机采样
                print(f"⚠️ FPS 采样失败，退化为无放回随机采样: {e}")
                indices = np.random.choice(current_num, target_num, replace=False)
                pcd_fixed = pcd[indices]
        else:
            pcd_fixed = pcd
            
        obs['point_cloud'] = pcd_fixed
    else:
        # 如果没用 DINOv2，保持原样
        obs['point_cloud'] = pcd
        
    # 3. ====== 终极防护网：清洗空值，转换 list (保留你原本优秀的防崩溃逻辑) ======
    keys_to_delete = []
    for key, val in list(obs.items()):  
        if isinstance(val, list):
            if len(val) == 0:
                keys_to_delete.append(key)  
            else:
                obs[key] = np.array(val, dtype=np.float32)
        elif isinstance(val, np.ndarray) and val.size == 0:
            keys_to_delete.append(key)      
            
    for key in keys_to_delete:
        del obs[key]  
    # =======================================================================
    
    return obs


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {"override_dirname": usr_args['task_name']},
        "run": {"dir": run_dir},
        "sweep": {"dir": run_dir, "subdir": "0"}
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    
    # =================================================================
    # 🌟 动态加载 DINOv2 大脑
    # 通过 usr_args (来自 deploy_policy.yml) 决定是否挂载 DINOv2
    # =================================================================
    if usr_args.get('use_semantic_mask', False):
        head_path = usr_args.get('dinov2_head_path', 'checkpoints/dinov2_linear_head.pth')
        # 将实例化的提取器作为属性挂载在 DP3_Model 上
        DP3_Model.semantic_extractor = SemanticPointExtractor(head_weight_path=head_path)
        print("👁️ DINOv2 语义过滤基座已成功挂载！")
    else:
        DP3_Model.semantic_extractor = None

    return DP3_Model


def eval(TASK_ENV, model, observation):
    #print(observation['observation']['rgb'].keys())
    # =================================================================
    # 🌟 首次观测拦截：覆盖传入的初始无掩膜 observation
    # =================================================================
    if getattr(model, 'semantic_extractor', None) is not None:
        
        rgb_img = observation['observation']['head_camera']['rgb'] 
        
        semantic_mask = model.semantic_extractor.predict(rgb_img)
        # 强制环境使用 DINOv2 的掩膜重新生成点云 (需要在 camera.py 里配合修改)
        observation = TASK_ENV.get_obs(semantic_mask=semantic_mask)
        
    obs = encode_obs(observation, model)  
    
    if len(model.env_runner.obs) == 0:  
        model.update_obs(obs)
    
    actions = model.get_action() 

    for action in actions:  
        TASK_ENV.take_action(action)
        # =================================================================
        # 🕵️‍♂️ 调试雷达：截获并保存最终喂给 DP3 的点云
        # =================================================================
        import os
        save_dir = "debug_pcd"
        os.makedirs(save_dir, exist_ok=True)
        
        if not hasattr(model, 'debug_frame_count'):
            model.debug_frame_count = 0
            
        # 为了不把硬盘写爆，我们只存前 100 帧，或者你可以自己加条件
        # 比如：只在机械臂高度低于某个值（靠近可乐时）才开始存
        if model.debug_frame_count < 300:
            # obs['point_cloud'] 的形状应该是 (1024, 6) -> XYZ + RGB
            pcd_to_save = obs['point_cloud']
            save_path = os.path.join(save_dir, f"frame_{model.debug_frame_count:03d}.npy")
            np.save(save_path, pcd_to_save)
            print(f"📸 抓拍点云已保存: {save_path}")
            
        model.debug_frame_count += 1
        # =================================================================
        # =================================================================
        # 🌟 动作循环观测拦截：获取带掩膜的新一帧
        # =================================================================
        if getattr(model, 'semantic_extractor', None) is not None:
            rgb_img = observation['observation']['head_camera']['rgb'] 
        
            semantic_mask = model.semantic_extractor.predict(rgb_img)
            observation = TASK_ENV.get_obs(semantic_mask=semantic_mask)
        else:
            observation = TASK_ENV.get_obs()
            
        obs = encode_obs(observation, model)
        model.update_obs(obs)  


def reset_model(model):  
    model.env_runner.reset_obs()