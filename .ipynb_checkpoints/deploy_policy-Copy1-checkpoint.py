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
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *


def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    return DP3_Model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()
    
    # ====== 终极防护网 1：清洗空值，转换 list ======
    keys_to_delete = []
    for key, val in list(obs.items()):  # 用 list() 包裹，防止遍历时字典大小改变报错
        if isinstance(val, list):
            if len(val) == 0:
                keys_to_delete.append(key)  # 发现空列表，记入暗杀名单
            else:
                obs[key] = np.array(val, dtype=np.float32)
        elif isinstance(val, np.ndarray) and val.size == 0:
            keys_to_delete.append(key)      # 发现空 numpy 数组，记入暗杀名单
            
    for key in keys_to_delete:
        del obs[key]  # 直接从字典中删除这些没用的空数据，不让它们进模型
    # ===============================================
    
    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)


    
    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        # ====== 终极防护网 2：清洗动作循环中的每一帧空值 ======
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
        # ======================================================
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
