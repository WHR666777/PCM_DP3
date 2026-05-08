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

    tide_chunk_failing = False
    if getattr(model, "tide_enabled", False):
        if action_meta is None or "action_pred" not in action_meta:
            raise RuntimeError("DP-TIDE enabled but action_pred metadata is unavailable.")
        tide_chunk_failing = update_tide_for_inference(model, action_meta, stride=actions.shape[0])

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
    ):
        if hasattr(model, attr):
            delattr(model, attr)
