# import packages and module here
import sys
import os
import numpy as np
import torch
import sapien.core as sapien
import traceback
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
sys.path.append(os.path.join(parent_directory, "3D-Diffusion-Policy"))

from dp3_policy import *

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from semantic_extractor import SemanticPointExtractor
except ImportError:
    SemanticPointExtractor = None
    print("WARNING: semantic_extractor.py not found; semantic mask will be unavailable.")

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        k = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
        else:
            points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=k)
        sampled_points = sampled_points.squeeze(0).detach().cpu().numpy()
        return sampled_points, indices

except Exception:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        raise RuntimeError("fps error: missing pytorch3d")


DBSCAN_EPS = 0.04
DBSCAN_MIN_POINTS = 150
TARGET_POINT_NUM = 1024


def decode_rgb_image(image):
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray, np.bytes_)):
        if cv2 is None:
            return None
        encoded = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    arr = np.asarray(image)
    if arr.ndim == 0 and arr.dtype.kind in ("S", "O"):
        return decode_rgb_image(arr.item())
    if arr.ndim == 2:
        if cv2 is None:
            return np.repeat(arr[:, :, None], 3, axis=2)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        max_value = np.nanmax(arr) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.copy()
    return arr


def get_head_camera_rgb(observation):
    return observation.get("observation", {}).get("head_camera", {}).get("rgb")


def filter_two_objects_dbscan(pcd, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS):
    if pcd.shape[0] == 0 or o3d is None:
        return pcd

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        if pcd.shape[0] > 0:
            print("WARNING: DBSCAN found no valid clusters; using masked point cloud.")
        return pcd

    counts = np.bincount(valid_labels)
    top_cluster_ids = np.argsort(counts)[-2:]
    keep = np.isin(labels, top_cluster_ids)
    return pcd[keep]


def fit_point_count(pcd, target_num=TARGET_POINT_NUM):
    pcd = np.asarray(pcd, dtype=np.float32)
    if pcd.ndim != 2:
        raise ValueError(f"pointcloud must be 2D, got shape {pcd.shape}")
    if pcd.shape[1] < 3:
        raise ValueError(f"pointcloud must have at least XYZ columns, got shape {pcd.shape}")

    if pcd.shape[1] < 6:
        padding = np.zeros((pcd.shape[0], 6 - pcd.shape[1]), dtype=np.float32)
        pcd = np.concatenate([pcd, padding], axis=1)
    elif pcd.shape[1] > 6:
        pcd = pcd[:, :6]

    current_num = pcd.shape[0]
    if current_num == 0:
        return np.zeros((target_num, 6), dtype=np.float32)
    if current_num == target_num:
        return pcd.astype(np.float32, copy=False)

    if current_num > target_num:
        try:
            _, indices_tensor = fps(pcd[:, :3], target_num, use_cuda=True)
            indices = indices_tensor.detach().cpu().numpy()[0]
        except Exception as e:
            if not hasattr(fit_point_count, "_printed_fps_warning"):
                print(f"WARNING: FPS failed once, falling back to random sampling: {e}")
                fit_point_count._printed_fps_warning = True
            indices = np.random.choice(current_num, target_num, replace=False)
    else:
        indices = np.random.choice(current_num, target_num, replace=True)
    return pcd[indices].astype(np.float32, copy=False)


def clean_empty_obs(obs):
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
    return obs


def encode_obs(observation, model=None):
    obs = dict()
    obs["agent_pos"] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

    raw_pcd = np.asarray(observation["pointcloud"], dtype=np.float32)
    eps = float(getattr(model, "dbscan_eps", DBSCAN_EPS))
    min_points = int(getattr(model, "dbscan_min_points", DBSCAN_MIN_POINTS))
    target_num = int(getattr(model, "target_point_num", TARGET_POINT_NUM))

    if raw_pcd.shape[0] > 0:
        pcd = filter_two_objects_dbscan(raw_pcd, eps=eps, min_points=min_points)
    else:
        pcd = raw_pcd
    obs["point_cloud"] = fit_point_count(pcd, target_num=target_num)
    return clean_empty_obs(obs)


def get_semantic_observation(TASK_ENV, model, observation):
    if getattr(model, "semantic_extractor", None) is None:
        return observation

    rgb_img = decode_rgb_image(get_head_camera_rgb(observation))
    if rgb_img is None:
        if not getattr(model, "printed_semantic_rgb_warning", False):
            print("WARNING: semantic mask skipped because head_camera/rgb is missing or invalid.")
            model.printed_semantic_rgb_warning = True
        return observation

    semantic_mask = model.semantic_extractor.predict(rgb_img)
    return TASK_ENV.get_obs(semantic_mask=semantic_mask)


def maybe_print_shapes(model, obs, actions=None):
    if not getattr(model, "printed_deploy_shapes", False):
        print(f"agent_pos shape: {obs['agent_pos'].shape}")
        print(f"point_cloud shape: {obs['point_cloud'].shape}")
        if actions is not None:
            print(f"action shape: {actions.shape}")
            model.printed_deploy_shapes = True


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base="1.2"):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {"override_dirname": usr_args["task_name"]},
        "run": {"dir": run_dir},
        "sweep": {"dir": run_dir, "subdir": "0"},
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args["use_rgb"]
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    DP3_Model.dbscan_eps = float(usr_args.get("dbscan_eps", DBSCAN_EPS))
    DP3_Model.dbscan_min_points = int(usr_args.get("dbscan_min_points", DBSCAN_MIN_POINTS))
    DP3_Model.target_point_num = int(usr_args.get("target_point_num", TARGET_POINT_NUM))

    if usr_args.get("use_semantic_mask", False):
        if SemanticPointExtractor is None:
            raise ImportError("use_semantic_mask=true but semantic_extractor.py could not be imported.")
        head_path = usr_args.get("dinov2_head_path", "checkpoints/dinov2_linear_head.pth")
        DP3_Model.semantic_extractor = SemanticPointExtractor(head_weight_path=head_path)
        print(f"DINOv2 semantic mask enabled: {head_path}")
    else:
        DP3_Model.semantic_extractor = None

    print(
        "Perception-filter-only DP3 deploy: "
        f"semantic={DP3_Model.semantic_extractor is not None}, "
        f"dbscan_eps={DP3_Model.dbscan_eps}, "
        f"dbscan_min_points={DP3_Model.dbscan_min_points}, "
        f"target_point_num={DP3_Model.target_point_num}"
    )
    return DP3_Model


def eval(TASK_ENV, model, observation):
    observation = get_semantic_observation(TASK_ENV, model, observation)
    obs = encode_obs(observation, model)

    if len(model.env_runner.obs) == 0:
        model.update_obs(obs)

    actions = model.get_action()
    maybe_print_shapes(model, obs, actions)

    for action in actions:
        TASK_ENV.take_action(action)
        raw_observation = TASK_ENV.get_obs()
        observation = get_semantic_observation(TASK_ENV, model, raw_observation)
        obs = encode_obs(observation, model)
        model.update_obs(obs)


def reset_model(model):
    model.env_runner.reset_obs()
    for attr in (
        "printed_deploy_shapes",
        "printed_semantic_rgb_warning",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
