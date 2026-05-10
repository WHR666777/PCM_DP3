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
import atexit
import csv
import json
import math
import re

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

import open3d as o3d
import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

DBSCAN_EPS = 0.04
DBSCAN_MIN_POINTS = 150
TASK_CENTER_FRAMES = 10
TARGET_POINT_NUM = 1024
DEBUG_VIDEO_MAX_FRAMES = 300
DEBUG_VIDEO_FPS = 20
DEBUG_VIDEO_PATH = os.path.join("debug_pcd", "debug_mask_pcd.mp4")
DEBUG_MASK_PANEL_SIZE = (640, 480)
DEBUG_PCD_PANEL_SIZE = (640, 480)
# Option 2: keep EE control but freeze orientation to reduce near-object planning jitter.
EE_LOCK_ROTATION = False
TIDE_OUTPUT_DIR = os.path.join("debug_pcd", "tide")
TIDE_THRESHOLD_PATH = os.path.join(TIDE_OUTPUT_DIR, "tide_threshold.json")
TIDE_MIN_CONSECUTIVE_FAIL = 2
REWIND_CHECKPOINT_DB_PATH = os.path.join("debug_pcd", "rewind", "rewind_checkpoint_db.npz")
REWIND_OUTPUT_DIR = os.path.join("debug_pcd", "rewind")
REWIND_PEAK_GAP = 2
REWIND_MIN_SIMILARITY = 0.55
REWIND_MIN_LIKELIHOOD = -1.0e18
REWIND_SLOT_ORDER = ["pre_reach", "pre_grasp", "grasp_ready", "pre_place", "lift_ready"]
REWIND_MIN_STAGE_DWELL = 2
REWIND_LIKELIHOOD_MARGIN = 0.05
REWIND_ALLOW_STAGE_SKIP = 1
REWIND_MIN_CONSECUTIVE_FAIL = 2
REWIND_COOLDOWN_STEPS = 12
REWIND_MAX_RECOVERY_ATTEMPTS = 1
REWIND_RECOVERY_INTERP_STEPS = 4
REWIND_WARMUP_OBS_STEPS = 3
REWIND_FORCE_RECOVERY = False
REWIND_FORCE_RECOVERY_ENV_STEP = -1
REWIND_FORCE_RECOVERY_INFER_IDX = -1
REWIND_FORCE_RECOVERY_SLOT = ""
REWIND_FORCE_RECOVERY_ONCE = True


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_slerp(q0, q1, t):
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return normalize_quat(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12)
    s1 = np.sin(theta) / max(sin_theta_0, 1e-12)
    return normalize_quat(s0 * q0 + s1 * q1)


def quat_to_matrix(q):
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(mat):
    mat = np.asarray(mat, dtype=np.float64)
    trace = np.trace(mat)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(mat)))
        if axis == 0:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            qw = (mat[2, 1] - mat[1, 2]) / s
            qx = 0.25 * s
            qy = (mat[0, 1] + mat[1, 0]) / s
            qz = (mat[0, 2] + mat[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            qw = (mat[0, 2] - mat[2, 0]) / s
            qx = (mat[0, 1] + mat[1, 0]) / s
            qy = 0.25 * s
            qz = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            qw = (mat[1, 0] - mat[0, 1]) / s
            qx = (mat[0, 2] + mat[2, 0]) / s
            qy = (mat[1, 2] + mat[2, 1]) / s
            qz = 0.25 * s
    return normalize_quat(np.array([qw, qx, qy, qz], dtype=np.float64))


def matrix_to_rot6d(mat):
    return np.concatenate([mat[:, 0], mat[:, 1]], axis=0)


def rot6d_to_matrix(rot6d):
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    if np.linalg.norm(a1) < 1e-6:
        a1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    a2_orth = a2 - np.dot(b1, a2) * b1
    if np.linalg.norm(a2_orth) < 1e-6:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, b1)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a2_orth = helper - np.dot(b1, helper) * b1
    b2 = a2_orth / max(np.linalg.norm(a2_orth), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def get_endpose(observation, arm):
    endpose = observation.get("endpose", {})
    key = f"{arm}_endpose"
    if key not in endpose:
        raise KeyError(f"observation['endpose']['{key}'] is required for cartesian DP3")
    return np.asarray(endpose[key], dtype=np.float64)


def get_gripper(observation, arm):
    endpose = observation.get("endpose", {})
    key = f"{arm}_gripper"
    if key in endpose:
        return float(np.asarray(endpose[key]).reshape(-1)[0])
    joint_action = observation.get("joint_action", {})
    if key in joint_action:
        return float(np.asarray(joint_action[key]).reshape(-1)[0])
    return 0.0


def fit_point_count(pcd, target_num=TARGET_POINT_NUM):
    if pcd.shape[0] == 0:
        return np.zeros((target_num, 6), dtype=np.float32)
    if pcd.shape[0] == target_num:
        return pcd.astype(np.float32, copy=False)
    if pcd.shape[0] > target_num:
        try:
            pcd_xyz = pcd[:, :3]
            _, indices_tensor = fps(pcd_xyz, target_num, use_cuda=True)
            indices = indices_tensor.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"FPS failed, fallback to random sampling: {e}")
            indices = np.random.choice(pcd.shape[0], target_num, replace=False)
        return pcd[indices].astype(np.float32, copy=False)
    indices = np.random.choice(pcd.shape[0], target_num, replace=True)
    return pcd[indices].astype(np.float32, copy=False)


def dbscan_two_object_result(pcd, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS):
    if pcd.shape[0] == 0:
        return pcd, np.zeros(3, dtype=np.float32), 0

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid_labels = labels[labels >= 0]

    if len(valid_labels) == 0:
        return pcd, pcd[:, :3].mean(axis=0).astype(np.float32), 0

    counts = np.bincount(valid_labels)
    top_cluster_ids = np.argsort(counts)[-2:]
    mask = np.isin(labels, top_cluster_ids)
    centers = np.asarray([pcd[labels == cluster_id, :3].mean(axis=0) for cluster_id in top_cluster_ids])
    task_center = centers.mean(axis=0).astype(np.float32)
    return pcd[mask], task_center, len(top_cluster_ids)


def get_head_camera_observation(observation):
    return observation.get("observation", {}).get("head_camera", {})


def decode_rgb_image(image):
    if image is None:
        return None

    if isinstance(image, (bytes, bytearray, np.bytes_)):
        encoded = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    arr = np.asarray(image)
    if arr.ndim == 0 and arr.dtype.kind in ("S", "O"):
        return decode_rgb_image(arr.item())
    if arr.ndim == 2:
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


def get_head_camera_matrices(observation):
    camera = get_head_camera_observation(observation)
    intrinsic = camera.get("intrinsic_cv", camera.get("intrinsic"))
    extrinsic = camera.get("extrinsic_cv", camera.get("extrinsic"))
    if intrinsic is None or extrinsic is None:
        raise KeyError("head_camera intrinsic_cv/extrinsic_cv are required for same-frame mask filtering")

    intrinsic = np.asarray(intrinsic, dtype=np.float64).squeeze()
    extrinsic = np.asarray(extrinsic, dtype=np.float64).squeeze()

    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    elif intrinsic.size == 9:
        intrinsic = intrinsic.reshape(3, 3)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")

    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    elif extrinsic.size == 16:
        extrinsic = extrinsic.reshape(4, 4)[:3, :]
    elif extrinsic.size == 12:
        extrinsic = extrinsic.reshape(3, 4)
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")

    return intrinsic, extrinsic


def filter_pointcloud_by_same_frame_mask(observation, semantic_mask):
    pcd = np.asarray(observation["pointcloud"])
    if pcd.shape[0] == 0:
        return pcd.copy(), {
            "mode": "same_frame_projection",
            "raw_points": 0,
            "projected_points": 0,
            "kept_points": 0,
        }

    intrinsic, extrinsic = get_head_camera_matrices(observation)
    mask = np.asarray(semantic_mask)
    height, width = mask.shape[:2]
    target_mask = (mask == 1) | (mask == 2)

    points = pcd[:, :3].astype(np.float64, copy=False)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    points_cam = (extrinsic @ points_homo.T).T
    points_2d_homo = (intrinsic @ points_cam.T).T

    depth = points_2d_homo[:, 2]
    valid_depth = (depth > 1e-5) & (points_cam[:, 2] > 1e-5)
    safe_depth = np.clip(depth, a_min=1e-5, a_max=None)
    u = (points_2d_homo[:, 0] / safe_depth).astype(np.int64)
    v = (points_2d_homo[:, 1] / safe_depth).astype(np.int64)

    valid_projection = (
        valid_depth
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    keep = np.zeros(pcd.shape[0], dtype=bool)
    valid_indices = np.where(valid_projection)[0]
    if valid_indices.size == 0:
        raise ValueError("no point projects into the head_camera image")
    if valid_indices.size > 0:
        keep[valid_indices] = target_mask[v[valid_indices], u[valid_indices]]

    filtered = pcd[keep].copy()
    stats = {
        "mode": "same_frame_projection",
        "raw_points": int(pcd.shape[0]),
        "projected_points": int(valid_projection.sum()),
        "kept_points": int(keep.sum()),
    }
    return filtered, stats


def make_mask_overlay(rgb_img, semantic_mask, output_size=DEBUG_MASK_PANEL_SIZE):
    if rgb_img is None:
        panel = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        cv2.putText(panel, "No RGB image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return panel

    rgb = decode_rgb_image(rgb_img)
    if rgb is None:
        panel = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        cv2.putText(panel, "Bad RGB image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return panel

    overlay = rgb.copy()
    if semantic_mask is not None:
        mask = np.asarray(semantic_mask)
        color_layer = np.zeros_like(overlay)
        color_layer[mask == 1] = np.array([255, 70, 70], dtype=np.uint8)
        color_layer[mask == 2] = np.array([70, 150, 255], dtype=np.uint8)
        target = (mask == 1) | (mask == 2)
        overlay[target] = (0.55 * overlay[target] + 0.45 * color_layer[target]).astype(np.uint8)

    panel = cv2.resize(overlay, output_size, interpolation=cv2.INTER_AREA)
    cv2.putText(panel, "DINO mask overlay", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(panel, "class 1: red  class 2: blue", (16, output_size[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def normalize_point_colors(pcd):
    if pcd.shape[1] < 6:
        return np.tile(np.array([[30, 130, 250]], dtype=np.uint8), (pcd.shape[0], 1))
    colors = pcd[:, 3:6]
    if colors.size == 0:
        return np.tile(np.array([[30, 130, 250]], dtype=np.uint8), (pcd.shape[0], 1))
    colors = colors.astype(np.float64, copy=False)
    if np.nanmax(colors) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def draw_point_projection(canvas, pcd, rect, axes, limits, title):
    x0, y0, width, height = rect
    margin = 18
    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (210, 210, 210), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    if pcd.shape[0] == 0:
        return

    points = pcd[:, :3]
    finite = np.isfinite(points).all(axis=1)
    if pcd.shape[1] >= 6:
        zero_padding = (np.linalg.norm(points, axis=1) < 1e-8) & (np.linalg.norm(pcd[:, 3:6], axis=1) < 1e-8)
        finite = finite & (~zero_padding)
    points = points[finite]
    if points.shape[0] == 0:
        return
    colors = normalize_point_colors(pcd[finite])

    x_axis, y_axis = axes
    x_min, x_max, y_min, y_max = limits
    x_vals = points[:, x_axis]
    y_vals = points[:, y_axis]
    px = x0 + margin + (x_vals - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * margin)
    py = y0 + height - margin - (y_vals - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * margin)
    px = np.clip(px, x0 + margin, x0 + width - margin - 1).astype(np.int32)
    py = np.clip(py, y0 + margin, y0 + height - margin - 1).astype(np.int32)

    for x, y, color in zip(px, py, colors):
        cv2.circle(canvas, (int(x), int(y)), 2, color.tolist(), -1, lineType=cv2.LINE_AA)


def make_pointcloud_panel(pcd, stats=None, output_size=DEBUG_PCD_PANEL_SIZE):
    panel = np.full((output_size[1], output_size[0], 3), 245, dtype=np.uint8)
    pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 3:
        cv2.putText(panel, "Bad point cloud", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        return panel

    top_rect = (0, 0, output_size[0], output_size[1] // 2)
    bottom_rect = (0, output_size[1] // 2, output_size[0], output_size[1] // 2)
    draw_point_projection(panel, pcd, top_rect, axes=(0, 1), limits=(-0.65, 0.65, -0.65, 0.65), title="Model point cloud: XY")
    draw_point_projection(panel, pcd, bottom_rect, axes=(0, 2), limits=(-0.65, 0.65, -0.25, 0.65), title="Model point cloud: XZ")

    kept_text = ""
    if stats:
        kept_text = (
            f" raw={stats.get('raw_points', '?')}"
            f" proj={stats.get('projected_points', '?')}"
            f" kept={stats.get('kept_points', '?')}"
        )
    cv2.putText(panel, f"points={pcd.shape[0]}{kept_text}", (16, output_size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    if stats and stats.get("mode"):
        cv2.putText(panel, str(stats["mode"]), (16, output_size[1] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return panel


def build_debug_video_frame(observation, obs, model):
    rgb_img = observation.get("_semantic_rgb")
    if rgb_img is None:
        rgb_img = get_head_camera_observation(observation).get("rgb")
    semantic_mask = observation.get("_semantic_mask")
    stats = observation.get("_semantic_stats", {})
    mask_panel = make_mask_overlay(rgb_img, semantic_mask)
    pcd_panel = make_pointcloud_panel(obs["point_cloud"], stats)
    frame = np.concatenate([mask_panel, pcd_panel], axis=1)
    frame_idx = getattr(model, "debug_video_frame_count", 0)
    cv2.line(frame, (DEBUG_MASK_PANEL_SIZE[0], 0), (DEBUG_MASK_PANEL_SIZE[0], frame.shape[0] - 1), (255, 255, 255), 2)

    bar_h = 38
    cv2.rectangle(frame, (0, frame.shape[0] - bar_h), (frame.shape[1] - 1, frame.shape[0] - 1), (15, 15, 15), -1)
    cv2.putText(
        frame,
        f"frame={frame_idx:03d}",
        (14, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    tide_overlay = getattr(model, "tide_overlay", None)
    if tide_overlay is not None:
        tide_value = tide_overlay.get("tide", math.nan)
        q_hat = tide_overlay.get("q_hat", math.nan)
        is_failing = bool(tide_overlay.get("is_failing", False))
        infer_idx = int(tide_overlay.get("infer_idx", -1))
        overlap_len = int(tide_overlay.get("overlap_len", 0))
        tide_text = "nan" if np.isnan(tide_value) else f"{tide_value:.6f}"
        cv2.putText(
            frame,
            f"DP-TIDE infer={infer_idx:03d} L={overlap_len} tide={tide_text} q={q_hat:.6f}",
            (160, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        state_text = "FAIL" if is_failing else "OK"
        state_color = (255, 80, 80) if is_failing else (80, 230, 120)
        cv2.putText(frame, state_text, (frame.shape[1] - 90, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
        if is_failing:
            cv2.rectangle(frame, (1, 1), (frame.shape[1] - 2, frame.shape[0] - 2), (255, 60, 60), 4)

    rewind_overlay = getattr(model, "rewind_overlay", None)
    if rewind_overlay is not None:
        slot = rewind_overlay.get("best_slot", "none")
        score = float(rewind_overlay.get("best_score", math.nan))
        score_mode = rewind_overlay.get("score_mode", "score")
        peaked = rewind_overlay.get("peaked_slot", "none")
        stage = rewind_overlay.get("stage_slot", "none")
        status = rewind_overlay.get("status", "OK")
        score_text = "nan" if np.isnan(score) else f"{score:.3f}"
        status_color = (255, 200, 70)
        if status == "RECOVERING":
            status_color = (255, 80, 80)
        elif status == "COOLDOWN":
            status_color = (80, 190, 255)
        elif status == "OK":
            status_color = (80, 230, 120)
        rewind_text = f"Rewind slot={slot} stage={stage} {score_mode}={score_text} peaked={peaked}"
        text_x = 14
        text_y = frame.shape[0] - bar_h - 10
        text_size, _ = cv2.getTextSize(rewind_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
        box_x0 = max(0, text_x - 8)
        box_y0 = max(0, text_y - text_size[1] - 8)
        box_x1 = min(frame.shape[1] - 1, text_x + text_size[0] + 8)
        box_y1 = min(frame.shape[0] - 1, text_y + 8)
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (18, 18, 18), -1)
        frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
        cv2.putText(
            frame,
            rewind_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            frame,
            rewind_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
        )
        status_x = min(frame.shape[1] - 190, box_x1 + 18)
        cv2.putText(frame, status, (status_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(frame, status, (status_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)
    return frame


def finalize_debug_video(model):
    writer = getattr(model, "debug_video_writer", None)
    if writer is not None:
        writer.release()
        model.debug_video_writer = None
        video_path = getattr(model, "debug_video_path", DEBUG_VIDEO_PATH)
        print(f"debug video saved: {video_path}")


def append_debug_video_frame(model, observation, obs):
    frame_count = getattr(model, "debug_video_frame_count", 0)
    if frame_count >= DEBUG_VIDEO_MAX_FRAMES or getattr(model, "debug_video_failed", False):
        return

    frame = build_debug_video_frame(observation, obs, model)
    writer = getattr(model, "debug_video_writer", None)
    if writer is None:
        base_dir = os.path.dirname(DEBUG_VIDEO_PATH)
        os.makedirs(base_dir, exist_ok=True)
        frame_size = (frame.shape[1], frame.shape[0])
        episode_id = int(getattr(model, "tide_episode_idx", getattr(model, "debug_episode_idx", 0)))
        video_path = os.path.join(base_dir, f"debug_mask_pcd_episode{episode_id:04d}.mp4")
        video_path = make_non_overwrite_path(video_path)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), DEBUG_VIDEO_FPS, frame_size)
        if not writer.isOpened():
            writer.release()
            video_path = make_non_overwrite_path(os.path.splitext(video_path)[0] + ".avi")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), DEBUG_VIDEO_FPS, frame_size)
        if not writer.isOpened():
            writer.release()
            model.debug_video_failed = True
            print("failed to open debug video writer")
            return
        model.debug_video_writer = writer
        model.debug_video_path = video_path
        if not getattr(model, "debug_video_atexit_registered", False):
            atexit.register(finalize_debug_video, model)
            model.debug_video_atexit_registered = True
        print(f"writing debug video: {video_path}")

    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    model.debug_video_frame_count = frame_count + 1
    if model.debug_video_frame_count >= DEBUG_VIDEO_MAX_FRAMES:
        finalize_debug_video(model)


def load_tide_threshold(threshold_path):
    if not os.path.isfile(threshold_path):
        return None
    with open(threshold_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "q_hat" in obj:
            return float(obj["q_hat"])
        if "threshold" in obj:
            return float(obj["threshold"])
    if isinstance(obj, (int, float)):
        return float(obj)
    raise ValueError(f"Unsupported threshold JSON format in {threshold_path}")


def make_non_overwrite_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    idx = 1
    while True:
        cand = f"{stem}_{idx:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def _next_episode_index_by_scan(directory, prefix):
    if not os.path.isdir(directory):
        return 0
    max_idx = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)")
    for name in os.listdir(directory):
        base = os.path.splitext(name)[0]
        m = pattern.match(base)
        if m:
            try:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)
            except Exception:
                pass
    return max_idx + 1


def init_tide_runtime_if_needed(model):
    if hasattr(model, "tide_runtime_initialized"):
        return
    model.tide_runtime_initialized = True
    model.tide_output_dir = getattr(model, "tide_output_dir", TIDE_OUTPUT_DIR)
    os.makedirs(model.tide_output_dir, exist_ok=True)
    loaded_threshold = load_tide_threshold(model.tide_threshold_path)
    if loaded_threshold is None:
        model.tide_threshold = float("inf")
        model.tide_collect_only = True
        print(
            f"DP-TIDE threshold missing at {model.tide_threshold_path}. "
            "Entering collect-only mode (q_hat=inf): will log TIDE but never trigger FAIL."
        )
    else:
        model.tide_threshold = float(loaded_threshold)
        model.tide_collect_only = False
    model.tide_trace_rows = []
    model.tide_prev_action_pred = None
    model.tide_infer_idx = 0
    model.tide_env_step_count = 0
    model.tide_step_flags = []
    model.tide_overlay = {
        "tide": math.nan,
        "q_hat": model.tide_threshold,
        "is_failing": False,
        "infer_idx": -1,
        "overlap_len": 0,
    }
    if not hasattr(model, "tide_episode_idx_seeded"):
        model.tide_episode_idx = _next_episode_index_by_scan(model.tide_output_dir, "tide_trace_episode")
        model.tide_episode_idx_seeded = True
    else:
        model.tide_episode_idx = getattr(model, "tide_episode_idx", 0)
    if not hasattr(model, "debug_episode_idx"):
        model.debug_episode_idx = model.tide_episode_idx


def compute_dp_tide(prev_action_pred, curr_action_pred, stride):
    prev_action_pred = np.asarray(prev_action_pred, dtype=np.float64)
    curr_action_pred = np.asarray(curr_action_pred, dtype=np.float64)
    if prev_action_pred.ndim != 2 or curr_action_pred.ndim != 2:
        raise ValueError(f"action_pred must be 2D, got {prev_action_pred.shape}, {curr_action_pred.shape}")

    horizon_prev = prev_action_pred.shape[0]
    horizon_curr = curr_action_pred.shape[0]
    horizon = min(horizon_prev, horizon_curr)
    overlap_len = max(0, horizon - int(stride))
    if overlap_len <= 0:
        return math.nan, overlap_len, horizon

    prev_overlap = prev_action_pred[int(stride): int(stride) + overlap_len]
    curr_overlap = curr_action_pred[:overlap_len]
    if prev_overlap.shape != curr_overlap.shape:
        min_len = min(prev_overlap.shape[0], curr_overlap.shape[0])
        prev_overlap = prev_overlap[:min_len]
        curr_overlap = curr_overlap[:min_len]
    tide = float(np.mean((prev_overlap - curr_overlap) ** 2))
    return tide, overlap_len, horizon


def update_tide_for_inference(model, action_meta, stride):
    init_tide_runtime_if_needed(model)
    curr_pred = np.asarray(action_meta["action_pred"]).squeeze(0)
    if curr_pred.ndim != 2:
        curr_pred = curr_pred.reshape(curr_pred.shape[0], -1)

    tide = math.nan
    overlap_len = 0
    horizon = int(curr_pred.shape[0])
    is_failing = False

    if model.tide_prev_action_pred is not None:
        tide, overlap_len, horizon = compute_dp_tide(model.tide_prev_action_pred, curr_pred, stride=stride)
        if not np.isnan(tide):
            is_failing = bool(tide > model.tide_threshold)

    if overlap_len == 0 and not hasattr(model, "printed_tide_overlap_warning"):
        print("DP-TIDE warning: no natural overlap between consecutive chunks; tide is NaN and disabled.")
        model.printed_tide_overlap_warning = True

    row = {
        "episode_id": int(model.tide_episode_idx),
        "infer_idx": int(model.tide_infer_idx),
        "env_step_start": int(model.tide_env_step_count),
        "tide": float(tide) if not np.isnan(tide) else math.nan,
        "q_hat": float(model.tide_threshold),
        "is_failing": int(is_failing),
        "overlap_len": int(overlap_len),
        "H": int(horizon),
        "S": int(stride),
    }
    model.tide_trace_rows.append(row)
    model.tide_overlay = {
        "tide": tide,
        "q_hat": model.tide_threshold,
        "is_failing": is_failing,
        "infer_idx": model.tide_infer_idx,
        "overlap_len": overlap_len,
    }
    model.tide_prev_action_pred = curr_pred.copy()
    model.tide_infer_idx += 1
    return is_failing


def _merge_failure_segments(step_flags, min_consecutive_fail):
    segments = []
    start = None
    for i, flag in enumerate(step_flags):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            if i - start >= min_consecutive_fail:
                segments.append([int(start), int(i - 1)])
            start = None
    if start is not None and len(step_flags) - start >= min_consecutive_fail:
        segments.append([int(start), int(len(step_flags) - 1)])
    return segments


def _save_tide_curve(rows, q_hat, curve_path):
    if plt is None:
        print("matplotlib not available, skip tide curve rendering.")
        return
    valid_rows = [r for r in rows if not np.isnan(r["tide"])]
    if len(valid_rows) == 0:
        return
    xs = np.array([r["infer_idx"] for r in valid_rows], dtype=np.int32)
    ys = np.array([r["tide"] for r in valid_rows], dtype=np.float64)
    fail_mask = ys > q_hat
    plt.figure(figsize=(8.0, 3.8))
    plt.plot(xs, ys, color="#1f77b4", linewidth=1.6, label="DP-TIDE")
    plt.axhline(q_hat, color="#d62728", linestyle="--", linewidth=1.4, label=f"q_hat={q_hat:.6f}")
    if np.any(fail_mask):
        plt.scatter(xs[fail_mask], ys[fail_mask], color="#d62728", s=16, label="Fail trigger")
    plt.xlabel("Inference Index")
    plt.ylabel("TIDE (MSE over overlap)")
    plt.title("DP-TIDE Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=220)
    plt.close()


def finalize_tide_episode(model):
    if not hasattr(model, "tide_runtime_initialized"):
        return
    rows = getattr(model, "tide_trace_rows", [])
    if len(rows) == 0:
        return

    episode_id = int(getattr(model, "tide_episode_idx", 0))
    out_dir = getattr(model, "tide_output_dir", TIDE_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = make_non_overwrite_path(os.path.join(out_dir, f"tide_trace_episode{episode_id:04d}.csv"))
    curve_path = make_non_overwrite_path(os.path.join(out_dir, f"tide_curve_episode{episode_id:04d}.png"))
    summary_path = make_non_overwrite_path(os.path.join(out_dir, f"tide_summary_episode{episode_id:04d}.json"))
    segments_path = make_non_overwrite_path(os.path.join(out_dir, f"failure_segments_episode{episode_id:04d}.json"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_id", "infer_idx", "env_step_start", "tide", "q_hat", "is_failing", "overlap_len", "H", "S"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    q_hat = float(rows[0]["q_hat"])
    _save_tide_curve(rows, q_hat, curve_path)

    tide_values = np.array([r["tide"] for r in rows if not np.isnan(r["tide"])], dtype=np.float64)
    fail_count = int(np.sum([r["is_failing"] for r in rows]))
    first_fail_infer = None
    for row in rows:
        if row["is_failing"] == 1:
            first_fail_infer = int(row["infer_idx"])
            break
    summary = {
        "episode_id": episode_id,
        "num_inference_calls": len(rows),
        "num_valid_tide": int(tide_values.shape[0]),
        "num_fail_triggers": fail_count,
        "first_fail_infer_idx": first_fail_infer,
        "q_hat": q_hat,
        "tide_mean": float(np.mean(tide_values)) if tide_values.size > 0 else None,
        "tide_max": float(np.max(tide_values)) if tide_values.size > 0 else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    segments = _merge_failure_segments(
        getattr(model, "tide_step_flags", []),
        int(getattr(model, "tide_min_consecutive_fail", TIDE_MIN_CONSECUTIVE_FAIL)),
    )
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_id": episode_id,
                "min_consecutive_fail": int(getattr(model, "tide_min_consecutive_fail", TIDE_MIN_CONSECUTIVE_FAIL)),
                "segments": segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"DP-TIDE saved: {csv_path}")
    print(f"DP-TIDE curve: {curve_path}")
    print(f"DP-TIDE summary: {summary_path}")
    print(f"DP-TIDE segments: {segments_path}")


def _normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm < 1e-12:
        return feature
    return feature / norm


def load_rewind_checkpoint_db(db_path):
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Rewind checkpoint db not found: {db_path}")
    data = np.load(db_path, allow_pickle=True)
    required = ["slot_names", "features", "recovery_actions"]
    for key in required:
        if key not in data:
            raise KeyError(f"{db_path} missing required array: {key}")

    slot_names = np.asarray(data["slot_names"]).astype(str)
    features = np.asarray(data["features"], dtype=np.float64)
    recovery_actions = np.asarray(data["recovery_actions"], dtype=np.float64)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if recovery_actions.ndim != 2 or recovery_actions.shape[1] != 20:
        raise ValueError(f"recovery_actions must have shape (N, 20), got {recovery_actions.shape}")
    if slot_names.shape[0] != features.shape[0] or slot_names.shape[0] != recovery_actions.shape[0]:
        raise ValueError("Rewind db arrays length mismatch.")

    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(feature_norms, 1e-12, None)
    source_episodes = np.asarray(data["source_episodes"]).astype(str) if "source_episodes" in data else np.array([""] * len(slot_names))
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64) if "frame_indices" in data else np.full(len(slot_names), -1)
    db = {
        "slot_names": slot_names,
        "features": features,
        "recovery_actions": recovery_actions,
        "source_episodes": source_episodes,
        "frame_indices": frame_indices,
    }
    if "pca_mean" in data and "pca_components" in data and "pca_features" in data:
        db["pca_mean"] = np.asarray(data["pca_mean"], dtype=np.float64).reshape(-1)
        db["pca_components"] = np.asarray(data["pca_components"], dtype=np.float64)
        db["pca_features"] = np.asarray(data["pca_features"], dtype=np.float64)
        bandwidth = np.asarray(data["kde_bandwidth"], dtype=np.float64).reshape(-1)[0] if "kde_bandwidth" in data else 1.0
        db["kde_bandwidth"] = max(float(bandwidth), 1e-6)
        db["score_mode"] = "pca_kde"
    else:
        db["score_mode"] = "cosine"
    return db


def gaussian_kde_log_likelihood(query, samples, bandwidth):
    samples = np.asarray(samples, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(1, -1)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return -float("inf"), -1
    diff = samples - query
    sq_dist = np.sum(diff * diff, axis=1)
    log_kernel = -0.5 * sq_dist / max(float(bandwidth) ** 2, 1e-12)
    max_log = float(np.max(log_kernel))
    log_likelihood = max_log + math.log(float(np.mean(np.exp(log_kernel - max_log))))
    nearest_local_idx = int(np.argmax(log_kernel))
    return float(log_likelihood), nearest_local_idx


def rewind_score_slots(db, feature):
    if db.get("score_mode") == "pca_kde":
        if feature.shape[0] != db["pca_mean"].shape[0]:
            raise ValueError(
                f"Rewind feature dim mismatch: online={feature.shape[0]}, db={db['pca_mean'].shape[0]}"
            )
        z = (feature - db["pca_mean"]) @ db["pca_components"].T
        scores = {}
        sample_indices = {}
        for slot in sorted(set(db["slot_names"].tolist())):
            idx = np.where(db["slot_names"] == slot)[0]
            score, local_idx = gaussian_kde_log_likelihood(z, db["pca_features"][idx], db["kde_bandwidth"])
            scores[slot] = score
            sample_indices[slot] = int(idx[local_idx]) if local_idx >= 0 else -1
        return scores, sample_indices, "likelihood"

    sims = db["features"] @ _normalize_feature(feature)
    scores = {}
    sample_indices = {}
    for slot in sorted(set(db["slot_names"].tolist())):
        idx = np.where(db["slot_names"] == slot)[0]
        local_sims = sims[idx]
        local_best = int(np.argmax(local_sims))
        scores[slot] = float(local_sims[local_best])
        sample_indices[slot] = int(idx[local_best])
    return scores, sample_indices, "cosine"


def init_rewind_runtime_if_needed(model):
    if hasattr(model, "rewind_runtime_initialized"):
        return
    model.rewind_runtime_initialized = True
    model.rewind_output_dir = getattr(model, "rewind_output_dir", REWIND_OUTPUT_DIR)
    os.makedirs(model.rewind_output_dir, exist_ok=True)
    model.rewind_db = load_rewind_checkpoint_db(model.rewind_checkpoint_db_path)
    model.rewind_slot_state = {}
    db_slots = sorted(set(model.rewind_db["slot_names"].tolist()))
    configured_order = list(getattr(model, "rewind_slot_order", REWIND_SLOT_ORDER))
    ordered_slots = [slot for slot in configured_order if slot in db_slots]
    ordered_slots.extend([slot for slot in db_slots if slot not in ordered_slots])
    model.rewind_slot_order = ordered_slots
    for slot in db_slots:
        model.rewind_slot_state[slot] = {
            "best_score": -float("inf"),
            "best_infer_idx": -1,
            "best_template_idx": -1,
            "recovery_action20": None,
            "peaked": False,
            "peaked_at": -1,
        }
    if not hasattr(model, "rewind_episode_idx_seeded"):
        model.rewind_episode_idx = _next_episode_index_by_scan(model.rewind_output_dir, "rewind_events_episode")
        model.rewind_episode_idx_seeded = True
    model.rewind_events = []
    model.rewind_consecutive_fail = 0
    model.rewind_cooldown_remaining = 0
    model.rewind_recovery_attempts = 0
    model.rewind_confirmed_stage_idx = -1
    model.rewind_candidate_stage = None
    model.rewind_candidate_dwell = 0
    model.rewind_last_confirmed_slot = None
    model.rewind_overlay = {
        "best_slot": "none",
        "best_score": math.nan,
        "score_mode": model.rewind_db.get("score_mode", "cosine"),
        "peaked_slot": "none",
        "status": "OK",
    }
    print(
        f"Rewind enabled. db={model.rewind_checkpoint_db_path}, "
        f"slots={model.rewind_slot_order}, score={model.rewind_db.get('score_mode')}, output={model.rewind_output_dir}"
    )


def update_rewind_tracker(model, action_meta):
    init_rewind_runtime_if_needed(model)
    if action_meta is None or "obs_feature" not in action_meta:
        raise RuntimeError("Rewind enabled but obs_feature metadata is unavailable.")

    feature = _normalize_feature(np.asarray(action_meta["obs_feature"]).squeeze(0))
    db = model.rewind_db
    scores, sample_indices, score_name = rewind_score_slots(db, feature)
    infer_idx = int(getattr(model, "tide_infer_idx", 0))
    best_slot = max(scores, key=scores.get)
    best_score = float(scores[best_slot])

    for slot, state in model.rewind_slot_state.items():
        if slot not in scores:
            continue
        score = float(scores[slot])
        template_idx = int(sample_indices[slot])
        if score > state["best_score"]:
            state["best_score"] = score
            state["best_infer_idx"] = infer_idx
            state["best_template_idx"] = template_idx
            state["recovery_action20"] = db["recovery_actions"][template_idx].astype(np.float32)
            state["peaked"] = False
            state["peaked_at"] = -1
    stage_slot, stage_score = update_rewind_stage_machine(model, scores, score_name, infer_idx)
    peaked_slot = select_rewind_recovery_slot(model)
    model.rewind_overlay = {
        "best_slot": best_slot,
        "best_score": best_score,
        "score_mode": score_name,
        "peaked_slot": peaked_slot if peaked_slot is not None else "none",
        "stage_slot": stage_slot if stage_slot is not None else "none",
        "stage_score": stage_score,
        "status": "COOLDOWN" if getattr(model, "rewind_cooldown_remaining", 0) > 0 else "OK",
    }
    return best_slot, best_score, peaked_slot


def update_rewind_stage_machine(model, scores, score_name, infer_idx):
    order = list(getattr(model, "rewind_slot_order", []))
    if len(order) == 0:
        return None, math.nan

    confirmed_idx = int(getattr(model, "rewind_confirmed_stage_idx", -1))
    allow_skip = max(0, int(getattr(model, "rewind_allow_stage_skip", REWIND_ALLOW_STAGE_SKIP)))
    start_idx = max(0, confirmed_idx)
    end_idx = min(len(order) - 1, confirmed_idx + allow_skip if confirmed_idx >= 0 else allow_skip)
    allowed = [slot for slot in order[start_idx:end_idx + 1] if slot in scores]
    if len(allowed) == 0:
        allowed = [slot for slot in order if slot in scores]

    stage_slot = max(allowed, key=lambda slot: scores[slot])
    stage_score = float(scores[stage_slot])
    stage_idx = order.index(stage_slot)

    if confirmed_idx >= 0 and stage_idx > confirmed_idx:
        current_slot = order[confirmed_idx]
        current_score = float(scores.get(current_slot, -float("inf")))
        if stage_score < current_score + float(getattr(model, "rewind_likelihood_margin", REWIND_LIKELIHOOD_MARGIN)):
            stage_slot = current_slot
            stage_score = current_score
            stage_idx = confirmed_idx

    if not rewind_score_passes_threshold(model, stage_score, score_name):
        return stage_slot, stage_score

    if getattr(model, "rewind_candidate_stage", None) == stage_slot:
        model.rewind_candidate_dwell += 1
    else:
        model.rewind_candidate_stage = stage_slot
        model.rewind_candidate_dwell = 1

    min_dwell = max(1, int(getattr(model, "rewind_min_stage_dwell", REWIND_MIN_STAGE_DWELL)))
    if model.rewind_candidate_dwell >= min_dwell and stage_idx > confirmed_idx:
        model.rewind_confirmed_stage_idx = stage_idx
        model.rewind_last_confirmed_slot = stage_slot
        state = model.rewind_slot_state[stage_slot]
        state["peaked"] = True
        state["peaked_at"] = infer_idx
        log_rewind_event(model, "stage_confirmed", selected_slot=stage_slot, score=stage_score)
    return stage_slot, stage_score


def rewind_score_passes_threshold(model, score, score_name):
    if score_name == "likelihood":
        return float(score) >= float(getattr(model, "rewind_min_likelihood", REWIND_MIN_LIKELIHOOD))
    return float(score) >= float(getattr(model, "rewind_min_similarity", REWIND_MIN_SIMILARITY))


def select_rewind_recovery_slot(model):
    slot = getattr(model, "rewind_last_confirmed_slot", None)
    if slot is None:
        return None
    state = getattr(model, "rewind_slot_state", {}).get(slot)
    if state is None or state.get("recovery_action20") is None:
        return None
    return slot


def ensure_rewind_slot_action(model, slot):
    if slot is None or slot == "":
        return None
    state = getattr(model, "rewind_slot_state", {}).get(slot)
    if state is None:
        return None
    if state.get("recovery_action20") is not None:
        return slot
    db = getattr(model, "rewind_db", None)
    if db is None:
        return None
    idx = np.where(db["slot_names"] == slot)[0]
    if idx.size == 0:
        return None
    template_idx = int(idx[0])
    state["best_template_idx"] = template_idx
    state["recovery_action20"] = db["recovery_actions"][template_idx].astype(np.float32)
    if not np.isfinite(state.get("best_score", -float("inf"))):
        state["best_score"] = 0.0
    return slot


def log_rewind_event(model, event_type, **kwargs):
    if not getattr(model, "rewind_enabled", False):
        return
    row = {
        "episode_id": int(getattr(model, "rewind_episode_idx", getattr(model, "tide_episode_idx", 0))),
        "infer_idx": int(getattr(model, "tide_infer_idx", -1)),
        "env_step": int(getattr(model, "tide_env_step_count", 0)),
        "event_type": event_type,
        "selected_slot": kwargs.pop("selected_slot", ""),
        "score": kwargs.pop("score", kwargs.pop("similarity", math.nan)),
        "score_mode": kwargs.pop("score_mode", getattr(model, "rewind_db", {}).get("score_mode", "")),
        "cooldown_remaining": int(getattr(model, "rewind_cooldown_remaining", 0)),
        "dry_run": int(bool(getattr(model, "rewind_dry_run", False))),
    }
    row.update(kwargs)
    model.rewind_events.append(row)


def should_force_rewind(model):
    if not getattr(model, "rewind_enabled", False) or not getattr(model, "rewind_force_recovery", False):
        return False, None
    if getattr(model, "rewind_force_recovery_once", True) and getattr(model, "rewind_force_recovery_used", False):
        return False, None
    env_step = int(getattr(model, "tide_env_step_count", 0))
    infer_idx = int(getattr(model, "tide_infer_idx", 0))
    force_env = int(getattr(model, "rewind_force_recovery_env_step", REWIND_FORCE_RECOVERY_ENV_STEP))
    force_infer = int(getattr(model, "rewind_force_recovery_infer_idx", REWIND_FORCE_RECOVERY_INFER_IDX))
    env_hit = force_env >= 0 and env_step >= force_env
    infer_hit = force_infer >= 0 and infer_idx >= force_infer
    if not env_hit and not infer_hit:
        return False, None
    requested_slot = str(getattr(model, "rewind_force_recovery_slot", "") or "")
    slot = ensure_rewind_slot_action(model, requested_slot) if requested_slot else select_rewind_recovery_slot(model)
    if slot is None:
        log_rewind_event(model, "force_no_recovery_target", selected_slot=requested_slot)
        model.rewind_force_recovery_used = bool(getattr(model, "rewind_force_recovery_once", True))
        return False, None
    model.rewind_force_recovery_used = True
    log_rewind_event(model, "force_recovery_trigger", selected_slot=slot)
    return True, slot


def should_trigger_rewind(model, tide_chunk_failing):
    if not getattr(model, "rewind_enabled", False):
        return False, None
    init_rewind_runtime_if_needed(model)
    if getattr(model, "rewind_cooldown_remaining", 0) > 0:
        model.rewind_cooldown_remaining -= 1
        model.rewind_overlay["status"] = "COOLDOWN"
        return False, None

    if tide_chunk_failing:
        model.rewind_consecutive_fail += 1
    else:
        model.rewind_consecutive_fail = 0
        return False, None

    if model.rewind_consecutive_fail < int(model.rewind_min_consecutive_fail):
        return False, None

    slot = select_rewind_recovery_slot(model)
    if slot is None:
        log_rewind_event(model, "no_recovery_target")
        return False, None
    state = model.rewind_slot_state[slot]
    score_name = "likelihood" if getattr(model, "rewind_db", {}).get("score_mode") == "pca_kde" else "cosine"
    if not rewind_score_passes_threshold(model, state["best_score"], score_name):
        log_rewind_event(model, "target_score_too_low", selected_slot=slot, score=float(state["best_score"]))
        return False, None
    return True, slot


def execute_rewind_recovery(TASK_ENV, model, observation, slot):
    state = model.rewind_slot_state[slot]
    recovery_action20 = np.asarray(state["recovery_action20"], dtype=np.float32)
    score = float(state["best_score"])
    log_rewind_event(model, "recover_start", selected_slot=slot, score=score)
    model.rewind_overlay["status"] = "RECOVERING"

    if getattr(model, "rewind_dry_run", False):
        print(f"Rewind dry-run: would recover to slot={slot}, score={score:.4f}")
        log_rewind_event(model, "recover_dry_run", selected_slot=slot, score=score)
        model.rewind_consecutive_fail = 0
        model.rewind_cooldown_remaining = int(model.rewind_cooldown_steps)
        return observation, encode_obs(observation, model)

    if int(getattr(model, "rewind_recovery_attempts", 0)) >= int(model.rewind_max_recovery_attempts):
        log_rewind_event(model, "unrecoverable", selected_slot=slot, score=score, reason="max_attempts")
        return observation, encode_obs(observation, model)

    interp_steps = max(1, int(getattr(model, "rewind_recovery_interp_steps", REWIND_RECOVERY_INTERP_STEPS)))
    target_ee_action = convert_model_action_to_ee_action(recovery_action20, observation, model.task_center)
    current_left = get_endpose(observation, "left")
    current_right = get_endpose(observation, "right")
    current_grippers = np.array([get_gripper(observation, "left"), get_gripper(observation, "right")], dtype=np.float64)

    for step_idx in range(interp_steps):
        alpha = float(step_idx + 1) / float(interp_steps)
        left_target = target_ee_action[:7].astype(np.float64)
        right_target = target_ee_action[8:15].astype(np.float64)
        interp_left = np.concatenate(
            [
                current_left[:3] + alpha * (left_target[:3] - current_left[:3]),
                quat_slerp(current_left[3:7], left_target[3:7], alpha),
            ],
            axis=0,
        )
        interp_right = np.concatenate(
            [
                current_right[:3] + alpha * (right_target[:3] - current_right[:3]),
                quat_slerp(current_right[3:7], right_target[3:7], alpha),
            ],
            axis=0,
        )
        interp_grippers = current_grippers + alpha * (
            np.array([target_ee_action[7], target_ee_action[15]], dtype=np.float64) - current_grippers
        )
        ee_action = np.concatenate(
            [
                interp_left,
                np.array([interp_grippers[0]], dtype=np.float64),
                interp_right,
                np.array([interp_grippers[1]], dtype=np.float64),
            ],
            axis=0,
        ).astype(np.float32)
        log_rewind_event(model, "recover_interp_step", selected_slot=slot, score=score, interp_step=step_idx + 1, interp_steps=interp_steps)
        TASK_ENV.take_action(ee_action, action_type="ee")

    raw_observation = TASK_ENV.get_obs()
    recovered_observation = get_semantic_observation(TASK_ENV, model, raw_observation)
    recovered_obs = encode_obs(recovered_observation, model)

    model.env_runner.reset_obs()
    warmup_steps = max(1, int(getattr(model, "rewind_warmup_obs_steps", REWIND_WARMUP_OBS_STEPS)))
    for warmup_idx in range(warmup_steps):
        if warmup_idx == 0:
            warm_observation = recovered_observation
            warm_obs = recovered_obs
        else:
            raw_warm_observation = TASK_ENV.get_obs()
            warm_observation = get_semantic_observation(TASK_ENV, model, raw_warm_observation)
            warm_obs = encode_obs(warm_observation, model)
        model.update_obs(warm_obs)
        recovered_observation = warm_observation
        recovered_obs = warm_obs
    model.tide_prev_action_pred = None
    model.rewind_consecutive_fail = 0
    model.rewind_cooldown_remaining = int(model.rewind_cooldown_steps)
    model.rewind_recovery_attempts += 1
    model.rewind_overlay["status"] = "COOLDOWN"
    log_rewind_event(model, "recover_done", selected_slot=slot, score=score, interp_steps=interp_steps, warmup_obs_steps=warmup_steps)
    print(f"Rewind recovered to slot={slot}, score={score:.4f}, interp_steps={interp_steps}, warmup_obs_steps={warmup_steps}")
    return recovered_observation, recovered_obs


def finalize_rewind_episode(model):
    if not getattr(model, "rewind_enabled", False) or not hasattr(model, "rewind_runtime_initialized"):
        return
    out_dir = getattr(model, "rewind_output_dir", REWIND_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    episode_id = int(getattr(model, "rewind_episode_idx", getattr(model, "tide_episode_idx", 0)))
    events = getattr(model, "rewind_events", [])
    events_path = make_non_overwrite_path(os.path.join(out_dir, f"rewind_events_episode{episode_id:04d}.csv"))
    summary_path = make_non_overwrite_path(os.path.join(out_dir, f"rewind_summary_episode{episode_id:04d}.json"))

    if len(events) > 0:
        fieldnames = sorted(set().union(*[set(e.keys()) for e in events]))
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow(event)

    slot_summary = {}
    for slot, state in getattr(model, "rewind_slot_state", {}).items():
        slot_summary[slot] = {
            "best_score": float(state.get("best_score", math.nan)),
            "best_infer_idx": int(state.get("best_infer_idx", -1)),
            "peaked": bool(state.get("peaked", False)),
            "peaked_at": int(state.get("peaked_at", -1)),
        }
    summary = {
        "episode_id": episode_id,
        "num_events": len(events),
        "num_recoveries": int(sum(1 for e in events if e.get("event_type") == "recover_done")),
        "dry_run": bool(getattr(model, "rewind_dry_run", False)),
        "slot_order": list(getattr(model, "rewind_slot_order", [])),
        "last_confirmed_slot": getattr(model, "rewind_last_confirmed_slot", None),
        "slots": slot_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Rewind summary: {summary_path}")
    if len(events) > 0:
        print(f"Rewind events: {events_path}")


def filter_two_objects_dbscan(pcd, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS):
    """
    使用 DBSCAN 保留点云中最大的两个簇（可乐和篮子），滤除稀疏杂物。
    pcd: np.ndarray, shape (N, 3) 或 (N, 6)
    """
    # 1. 转换为 Open3D 格式 (仅取前三维 XYZ 计算距离)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])

    # 2. 运行 DBSCAN
    # labels 中 -1 代表离群噪点，0, 1, 2... 代表不同的簇
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    # 3. 过滤掉 -1 (噪点)
    valid_labels = labels[labels >= 0]
    
    if len(valid_labels) == 0:
        # 极端异常情况：全是噪点，直接退化返回原数据
        print("⚠️ DBSCAN 未找到任何有效簇，返回原点云")
        return pcd

    # 4. 统计每个簇的点数，寻找 Top-2
    counts = np.bincount(valid_labels)
    # argsort 默认从小到大排，[-2:] 取最大的两个（如果只有一个簇，也能兼容取到那一个）
    top_cluster_ids = np.argsort(counts)[-2:] 

    # 5. 生成只包含这两个簇的布尔掩膜
    mask = np.isin(labels, top_cluster_ids)
    clean_pcd = pcd[mask]

    # # 6. 🚨 必须重采样补齐到 1024 个点 (为了满足 DP3 网络输入)
    # current_num = clean_pcd.shape[0]
    # if current_num < target_num:
    #     # 匮乏分支：有放回重采样补齐
    #     indices = np.random.choice(current_num, target_num, replace=True)
    #     clean_pcd = clean_pcd[indices]
    # elif current_num > target_num:
    #     # 如果传入的点数大于 1024 且过滤后仍大于 1024 (通常不会发生，安全兜底)
    #     indices = np.random.choice(current_num, target_num, replace=False)
    #     clean_pcd = clean_pcd[indices]

    return clean_pcd

# def encode_obs(observation, model=None):  
#     """ Post-Process Observation (包含零点坍缩修复) """
#     obs = dict()
#     obs['agent_pos'] = observation['joint_action']['vector']
    
#     # 1. 提取点云
#     pcd = observation['pointcloud']
    
#     # 2. 【核心修复】：解决 DP3 零点坍缩问题 (Point Starvation)
#     # 只有当启用了 DINOv2 时，点云才可能因为掩膜过滤而变得极其稀疏
#     if True:
#         target_num = 1024  # 必须与你 DP3 config 中的点云数完全一致
#         current_num = pcd.shape[0]
#         #print(current_num)
#         if current_num == 0:
#             print("⚠️ 警告：当前帧 DINOv2 未提取到任何目标！")
#             # 灾难兜底：避免 Numpy 报错，只能全零填充
#             pcd_fixed = np.zeros((target_num, pcd.shape[1]), dtype=np.float32)
            
#         elif current_num < target_num:
#             # 匮乏分支 -> 【有放回的随机重采样】 (避免补零导致 PointNet 重心偏移)
#             # indices = np.random.choice(current_num, target_num, replace=True)
#             # pcd_fixed = pcd[indices]
#             pad_num = target_num - current_num
#             padding = np.zeros((pad_num, pcd.shape[1]), dtype=np.float32)
#             pcd_fixed = np.vstack([pcd, padding])
            
#         elif current_num > target_num:
#             try:
#                 # 提取前三维 (X, Y, Z) 送去跑 FPS
#                 pcd_xyz = pcd[:, :3]
                
#                 # 调用你的 pytorch3d fps 接口
#                 _, indices_tensor = fps(pcd_xyz, target_num, use_cuda=True)
                
#                 # 解析返回的 tensor 索引
#                 indices = indices_tensor.detach().cpu().numpy()[0]
#                 pcd_fixed = pcd[indices]
#                 #print(pcd_fixed.shape[0])
                
#             except Exception as e:
#                 # 如果 pytorch3d 崩了或者没导入成功，作为兜底才使用随机采样
#                 print(f"⚠️ FPS 采样失败，退化为无放回随机采样: {e}")
#                 indices = np.random.choice(current_num, target_num, replace=False)
#                 pcd_fixed = pcd[indices]
#         else:
#             pcd_fixed = pcd
            
#         obs['point_cloud'] = pcd_fixed
#     else:
#         # 如果没用 DINOv2，保持原样
#         obs['point_cloud'] = pcd
        

    # # 3. ====== 终极防护网：清洗空值，转换 list (保留你原本优秀的防崩溃逻辑) ======
    # keys_to_delete = []
    # for key, val in list(obs.items()):  
    #     if isinstance(val, list):
    #         if len(val) == 0:
    #             keys_to_delete.append(key)  
    #         else:
    #             obs[key] = np.array(val, dtype=np.float32)
    #     elif isinstance(val, np.ndarray) and val.size == 0:
    #         keys_to_delete.append(key)      
            
    # for key in keys_to_delete:
    #     del obs[key]  
    # # =======================================================================
    
    # return obs

def legacy_encode_obs_unused(observation, model=None):  
    """ Post-Process Observation (严谨管线：Mask过滤 -> DBSCAN去噪 -> 补零/FPS对齐) """
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    # 1. 提取点云 (这里接收到的已经是经过 DINOv2 Mask 过滤后的点云)
    raw_pcd = observation['pointcloud']
    
    # 2. 【DBSCAN 剔除】：清理 Mask 漏网的桌面杂物 (保留可乐和篮子)
    if raw_pcd.shape[0] > 0:
        pcd = filter_two_objects_dbscan(raw_pcd, eps=0.04, min_points=150)
    else:
        pcd = raw_pcd

    # 3. 【数量规范化】：补零或 FPS，对齐 1024 维度
    if True: # 假设默认启用处理流
        target_num = 1024  # 必须与你 DP3 config 中的点云数完全一致
        current_num = pcd.shape[0]
        
        if current_num == 0:
            print("⚠️ 警告：当前帧未提取到任何目标！")
            # 灾难兜底：避免 Numpy 报错，全零填充
            pcd_fixed = np.zeros((target_num, pcd.shape[1]), dtype=np.float32)
            
        elif current_num < target_num:
            # 匮乏分支 -> 【补零】 (严丝合缝对齐训练数据分布)
            pad_num = target_num - current_num
            padding = np.zeros((pad_num, pcd.shape[1]), dtype=np.float32)
            pcd_fixed = np.vstack([pcd, padding])
            
        elif current_num > target_num:
            # 富裕分支 -> 【FPS 降采样】
            try:
                # 提取前三维 (X, Y, Z) 送去跑 FPS
                pcd_xyz = pcd[:, :3]
                
                # 调用 pytorch3d fps 接口
                _, indices_tensor = fps(pcd_xyz, target_num, use_cuda=True)
                
                # 解析返回的 tensor 索引
                indices = indices_tensor.detach().cpu().numpy()[0]
                pcd_fixed = pcd[indices]
                
            except Exception as e:
                print(f"⚠️ FPS 采样失败，退化为无放回随机采样: {e}")
                indices = np.random.choice(current_num, target_num, replace=False)
                pcd_fixed = pcd[indices]
        else:
            pcd_fixed = pcd
            
        obs['point_cloud'] = pcd_fixed
    else:
        # 如果没用 DINOv2，保持原样
        obs['point_cloud'] = raw_pcd
        
    return obs


def make_agent_pos(observation, task_center):
    left_pose = get_endpose(observation, "left")
    right_pose = get_endpose(observation, "right")
    left_rot6d = matrix_to_rot6d(quat_to_matrix(left_pose[3:7]))
    right_rot6d = matrix_to_rot6d(quat_to_matrix(right_pose[3:7]))
    left_state = np.concatenate(
        [
            left_pose[:3] - task_center,
            left_rot6d,
            np.array([get_gripper(observation, "left")], dtype=np.float64),
        ],
        axis=0,
    )
    right_state = np.concatenate(
        [
            right_pose[:3] - task_center,
            right_rot6d,
            np.array([get_gripper(observation, "right")], dtype=np.float64),
        ],
        axis=0,
    )
    return np.concatenate([left_state, right_state], axis=0).astype(np.float32)


def encode_obs(observation, model=None):
    if model is None or not hasattr(model, "task_center"):
        raise RuntimeError("model.task_center must be initialized before encode_obs")

    raw_pcd = observation["pointcloud"]
    pcd, _, _ = dbscan_two_object_result(raw_pcd, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS)
    pcd = pcd.astype(np.float32, copy=True)
    if pcd.shape[0] > 0:
        pcd[:, :3] -= model.task_center.astype(np.float32)

    return {
        "agent_pos": make_agent_pos(observation, model.task_center),
        "point_cloud": fit_point_count(pcd),
    }


def get_semantic_observation(TASK_ENV, model, observation):
    if getattr(model, "semantic_extractor", None) is None:
        return observation
    rgb_img = decode_rgb_image(get_head_camera_observation(observation).get("rgb"))
    if rgb_img is None:
        if not getattr(model, "printed_semantic_rgb_warning", False):
            print("semantic mask skipped: missing or invalid head_camera rgb")
            model.printed_semantic_rgb_warning = True
        return observation

    semantic_mask = model.semantic_extractor.predict(rgb_img)
    try:
        filtered_pcd, stats = filter_pointcloud_by_same_frame_mask(observation, semantic_mask)
        semantic_observation = dict(observation)
        semantic_observation["pointcloud"] = filtered_pcd
        semantic_observation["_semantic_rgb"] = rgb_img
        semantic_observation["_semantic_mask"] = semantic_mask
        semantic_observation["_semantic_stats"] = stats
        return semantic_observation
    except Exception as e:
        if not getattr(model, "printed_same_frame_filter_warning", False):
            print(f"same-frame mask filtering failed, falling back to TASK_ENV.get_obs(semantic_mask=...): {e}")
            model.printed_same_frame_filter_warning = True
        fallback_observation = TASK_ENV.get_obs(semantic_mask=semantic_mask)
        fallback_observation = dict(fallback_observation)
        fallback_observation["_semantic_rgb"] = rgb_img
        fallback_observation["_semantic_mask"] = semantic_mask
        fallback_observation["_semantic_stats"] = {
            "mode": "fallback_get_obs",
            "raw_points": int(np.asarray(observation.get("pointcloud", [])).shape[0]),
            "projected_points": 0,
            "kept_points": int(np.asarray(fallback_observation.get("pointcloud", [])).shape[0]),
            "error": str(e),
        }
        return fallback_observation


def initialize_task_center(TASK_ENV, model, observation, center_frames=TASK_CENTER_FRAMES):
    centers = []
    semantic_observation = get_semantic_observation(TASK_ENV, model, observation)

    for frame_idx in range(center_frames):
        _, task_center, cluster_count = dbscan_two_object_result(
            semantic_observation["pointcloud"],
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
        )
        if np.isfinite(task_center).all():
            centers.append(task_center)
        print(f"task center init {frame_idx + 1}/{center_frames}: {task_center}, clusters={cluster_count}")

        if frame_idx != center_frames - 1:
            raw_observation = TASK_ENV.get_obs()
            semantic_observation = get_semantic_observation(TASK_ENV, model, raw_observation)

    if len(centers) == 0:
        model.task_center = np.zeros(3, dtype=np.float32)
    else:
        model.task_center = np.median(np.asarray(centers, dtype=np.float32), axis=0).astype(np.float32)

    print(f"fixed task center: {model.task_center}")
    return semantic_observation


def convert_model_action_to_ee_action(action, observation, task_center):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 20:
        raise ValueError(f"Expected 20D cartesian action, got shape {action.shape}")

    task_center = np.asarray(task_center, dtype=np.float64).reshape(3)
    left_pose = get_endpose(observation, "left")
    right_pose = get_endpose(observation, "right")

    def convert_one(arm_action, current_pose):
        current_rot = quat_to_matrix(current_pose[3:7])
        target_pos = task_center + arm_action[:3]
        if EE_LOCK_ROTATION:
            target_rot = current_rot
        else:
            target_rot = rot6d_to_matrix(arm_action[3:9])
        target_quat = matrix_to_quat(target_rot)
        return np.concatenate([target_pos, target_quat], axis=0)

    left_target = convert_one(action[:10], left_pose)
    right_target = convert_one(action[10:20], right_pose)
    ee_action = np.concatenate(
        [
            left_target,
            np.array([action[9]], dtype=np.float64),
            right_target,
            np.array([action[19]], dtype=np.float64),
        ],
        axis=0,
    )
    return ee_action.astype(np.float32)


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

    DP3_Model.tide_enabled = bool(usr_args.get("tide_enabled", True))
    DP3_Model.tide_threshold_path = usr_args.get("tide_threshold_path", TIDE_THRESHOLD_PATH)
    DP3_Model.tide_output_dir = usr_args.get("tide_output_dir", TIDE_OUTPUT_DIR)
    DP3_Model.tide_min_consecutive_fail = int(usr_args.get("tide_min_consecutive_fail", TIDE_MIN_CONSECUTIVE_FAIL))
    if DP3_Model.tide_enabled:
        print(
            f"DP-TIDE enabled. threshold={DP3_Model.tide_threshold_path}, "
            f"output={DP3_Model.tide_output_dir}, min_fail={DP3_Model.tide_min_consecutive_fail}"
        )

    DP3_Model.rewind_enabled = bool(usr_args.get("rewind_enabled", False))
    DP3_Model.rewind_checkpoint_db_path = usr_args.get("rewind_checkpoint_db_path", REWIND_CHECKPOINT_DB_PATH)
    DP3_Model.rewind_output_dir = usr_args.get("rewind_output_dir", REWIND_OUTPUT_DIR)
    DP3_Model.rewind_peak_gap = int(usr_args.get("rewind_peak_gap", REWIND_PEAK_GAP))
    DP3_Model.rewind_min_similarity = float(usr_args.get("rewind_min_similarity", REWIND_MIN_SIMILARITY))
    DP3_Model.rewind_min_likelihood = float(usr_args.get("rewind_min_likelihood", REWIND_MIN_LIKELIHOOD))
    DP3_Model.rewind_slot_order = list(usr_args.get("rewind_slot_order", REWIND_SLOT_ORDER))
    DP3_Model.rewind_min_stage_dwell = int(usr_args.get("rewind_min_stage_dwell", REWIND_MIN_STAGE_DWELL))
    DP3_Model.rewind_likelihood_margin = float(usr_args.get("rewind_likelihood_margin", REWIND_LIKELIHOOD_MARGIN))
    DP3_Model.rewind_allow_stage_skip = int(usr_args.get("rewind_allow_stage_skip", REWIND_ALLOW_STAGE_SKIP))
    DP3_Model.rewind_min_consecutive_fail = int(usr_args.get("rewind_min_consecutive_fail", REWIND_MIN_CONSECUTIVE_FAIL))
    DP3_Model.rewind_cooldown_steps = int(usr_args.get("rewind_cooldown_steps", REWIND_COOLDOWN_STEPS))
    DP3_Model.rewind_max_recovery_attempts = int(usr_args.get("rewind_max_recovery_attempts", REWIND_MAX_RECOVERY_ATTEMPTS))
    DP3_Model.rewind_recovery_interp_steps = int(usr_args.get("rewind_recovery_interp_steps", REWIND_RECOVERY_INTERP_STEPS))
    DP3_Model.rewind_warmup_obs_steps = int(usr_args.get("rewind_warmup_obs_steps", REWIND_WARMUP_OBS_STEPS))
    DP3_Model.rewind_force_recovery = bool(usr_args.get("rewind_force_recovery", REWIND_FORCE_RECOVERY))
    DP3_Model.rewind_force_recovery_env_step = int(usr_args.get("rewind_force_recovery_env_step", REWIND_FORCE_RECOVERY_ENV_STEP))
    DP3_Model.rewind_force_recovery_infer_idx = int(usr_args.get("rewind_force_recovery_infer_idx", REWIND_FORCE_RECOVERY_INFER_IDX))
    DP3_Model.rewind_force_recovery_slot = str(usr_args.get("rewind_force_recovery_slot", REWIND_FORCE_RECOVERY_SLOT) or "")
    DP3_Model.rewind_force_recovery_once = bool(usr_args.get("rewind_force_recovery_once", REWIND_FORCE_RECOVERY_ONCE))
    DP3_Model.rewind_dry_run = bool(usr_args.get("rewind_dry_run", True))
    if DP3_Model.rewind_enabled:
        print(
            f"Rewind enabled. db={DP3_Model.rewind_checkpoint_db_path}, "
            f"dry_run={DP3_Model.rewind_dry_run}, min_like={DP3_Model.rewind_min_likelihood}"
        )

    return DP3_Model


def legacy_eval_unused(TASK_ENV, model, observation):
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
            #print(f"📸 抓拍点云已保存: {save_path}")
            
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


def legacy_reset_model_unused(model):  
    model.env_runner.reset_obs()


def eval(TASK_ENV, model, observation):
    if getattr(model, "tide_enabled", False):
        init_tide_runtime_if_needed(model)
    if getattr(model, "rewind_enabled", False):
        init_rewind_runtime_if_needed(model)

    if not hasattr(model, "task_center"):
        observation = initialize_task_center(TASK_ENV, model, observation)
    else:
        observation = get_semantic_observation(TASK_ENV, model, observation)

    obs = encode_obs(observation, model)
    if len(model.env_runner.obs) == 0:
        model.update_obs(obs)

    if hasattr(model, "get_action_with_meta"):
        actions, action_meta = model.get_action_with_meta()
    else:
        actions = model.get_action()
        action_meta = None

    if getattr(model, "rewind_enabled", False):
        update_rewind_tracker(model, action_meta)

    tide_chunk_failing = False
    if getattr(model, "tide_enabled", False):
        if action_meta is None or "action_pred" not in action_meta:
            raise RuntimeError("DP-TIDE enabled but action_pred metadata is unavailable.")
        tide_chunk_failing = update_tide_for_inference(model, action_meta, stride=actions.shape[0])

    if getattr(model, "rewind_enabled", False):
        do_recover, rewind_slot = should_force_rewind(model)
        if not do_recover:
            do_recover, rewind_slot = should_trigger_rewind(model, tide_chunk_failing)
        if do_recover:
            observation, obs = execute_rewind_recovery(TASK_ENV, model, observation, rewind_slot)
            if not getattr(model, "rewind_dry_run", False):
                return

    if not hasattr(model, "printed_action_shape"):
        print(f"model action shape: {actions.shape}")
        model.printed_action_shape = True

    for action in actions:
        append_debug_video_frame(model, observation, obs)
        ee_action = convert_model_action_to_ee_action(action, observation, model.task_center)
        if not hasattr(model, "printed_ee_action_shape"):
            print(f"converted ee action shape: {ee_action.shape}")
            model.printed_ee_action_shape = True
        TASK_ENV.take_action(ee_action, action_type="ee")

        raw_observation = TASK_ENV.get_obs()
        observation = get_semantic_observation(TASK_ENV, model, raw_observation)
        obs = encode_obs(observation, model)
        model.update_obs(obs)

        if getattr(model, "tide_enabled", False):
            model.tide_step_flags.append(bool(tide_chunk_failing))

    if getattr(model, "tide_enabled", False):
        model.tide_env_step_count += int(actions.shape[0])


def reset_model(model):
    if getattr(model, "rewind_enabled", False):
        finalize_rewind_episode(model)
        model.rewind_episode_idx = int(getattr(model, "rewind_episode_idx", getattr(model, "tide_episode_idx", 0))) + 1
    if getattr(model, "tide_enabled", False):
        finalize_tide_episode(model)
        model.tide_episode_idx = int(getattr(model, "tide_episode_idx", 0)) + 1
        model.debug_episode_idx = int(getattr(model, "debug_episode_idx", model.tide_episode_idx)) + 1
    finalize_debug_video(model)
    model.env_runner.reset_obs()
    for attr in (
        "task_center",
        "printed_action_shape",
        "printed_ee_action_shape",
        "debug_frame_count",
        "debug_video_writer",
        "debug_video_frame_count",
        "debug_video_path",
        "debug_video_failed",
        "debug_video_atexit_registered",
        "printed_semantic_rgb_warning",
        "printed_same_frame_filter_warning",
        "tide_runtime_initialized",
        "tide_trace_rows",
        "tide_prev_action_pred",
        "tide_infer_idx",
        "tide_env_step_count",
        "tide_step_flags",
        "tide_overlay",
        "printed_tide_overlap_warning",
        "tide_collect_only",
        "rewind_runtime_initialized",
        "rewind_db",
        "rewind_slot_state",
        "rewind_events",
        "rewind_consecutive_fail",
        "rewind_cooldown_remaining",
        "rewind_recovery_attempts",
        "rewind_overlay",
        "rewind_confirmed_stage_idx",
        "rewind_candidate_stage",
        "rewind_candidate_dwell",
        "rewind_last_confirmed_slot",
        "rewind_force_recovery_used",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
