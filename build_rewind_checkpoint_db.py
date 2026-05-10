import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
DP3_DIR = THIS_DIR / "3D-Diffusion-Policy"
sys.path.append(str(DP3_DIR))


def find_episode_path(data_dir, episode_name):
    candidates = [
        pathlib.Path(data_dir) / episode_name,
        pathlib.Path(data_dir) / "data" / episode_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {episode_name} under {data_dir}")


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "episodes" not in obj:
        raise KeyError("annotations JSON must contain 'episodes'")
    return obj


def load_model(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from dp3_policy import DP3

    config_dir = DP3_DIR / "diffusion_policy_3d" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        cfg = compose(config_name=f"{args.config_name}.yaml")

    OmegaConf.set_struct(cfg, False)
    cfg.task_name = args.task_name
    cfg.expert_data_num = args.expert_data_num
    cfg.raw_task_name = args.task_name
    cfg.policy.use_pc_color = args.use_rgb
    OmegaConf.set_struct(cfg, True)

    usr_args = {
        "task_name": args.task_name,
        "ckpt_setting": args.ckpt_setting,
        "expert_data_num": args.expert_data_num,
        "seed": args.seed,
        "checkpoint_num": args.checkpoint_num,
        "use_rgb": args.use_rgb,
    }
    old_cwd = os.getcwd()
    os.chdir(str(DP3_DIR))
    try:
        model = DP3(cfg, usr_args)
    finally:
        os.chdir(old_cwd)
    return model


def read_frame_obs(root, frame_idx):
    if "/state/vector" not in root or "/pointcloud" not in root:
        raise KeyError("hdf5 must contain /state/vector and /pointcloud")
    pointcloud = np.asarray(root["/pointcloud"][frame_idx], dtype=np.float32)
    agent_pos = np.asarray(root["/state/vector"][frame_idx], dtype=np.float32)
    recovery_action20 = agent_pos.copy()
    if recovery_action20.shape[0] != 20:
        raise ValueError(f"state/vector frame must be 20D, got {recovery_action20.shape}")
    return {"point_cloud": pointcloud, "agent_pos": agent_pos}, recovery_action20


def extract_feature(model, obs):
    model.env_runner.reset_obs()
    model.update_obs(obs)
    _, action_meta = model.get_action_with_meta()
    if "obs_feature" not in action_meta:
        raise RuntimeError("policy did not return obs_feature; update dp3.py first")
    feature = np.asarray(action_meta["obs_feature"]).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feature)
    if norm > 1e-12:
        feature = feature / norm
    return feature


def iter_slot_frames(slots):
    for slot_name, frame_value in slots.items():
        if isinstance(frame_value, list):
            frame_values = frame_value
        else:
            frame_values = [frame_value]
        for frame_idx in frame_values:
            yield slot_name, int(frame_idx)


def fit_pca(features, requested_dim):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    mean = features.mean(axis=0)
    centered = features - mean
    max_dim = max(1, min(int(requested_dim), centered.shape[0] - 1 if centered.shape[0] > 1 else 1, centered.shape[1]))
    if centered.shape[0] <= 1:
        components = np.zeros((max_dim, centered.shape[1]), dtype=np.float64)
        components[0, 0] = 1.0
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:max_dim]
    pca_features = centered @ components.T
    return mean.astype(np.float32), components.astype(np.float32), pca_features.astype(np.float32)


def estimate_bandwidth(pca_features, slot_names, user_bandwidth):
    if user_bandwidth is not None and user_bandwidth > 0:
        return float(user_bandwidth)
    pca_features = np.asarray(pca_features, dtype=np.float64)
    distances = []
    for slot in sorted(set(slot_names)):
        idx = [i for i, s in enumerate(slot_names) if s == slot]
        if len(idx) < 2:
            continue
        pts = pca_features[idx]
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=-1))
        tri = d[np.triu_indices(len(idx), k=1)]
        distances.extend([float(x) for x in tri if x > 1e-9])
    if len(distances) == 0:
        scale = float(np.std(pca_features)) if pca_features.size else 1.0
        return max(scale, 1e-3)
    return max(float(np.median(distances)), 1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing episode hdf5 files or data/episode*.hdf5.")
    parser.add_argument("--annotations", required=True, help="rewind_checkpoint_annotations.json")
    parser.add_argument("--out_npz", default=str(THIS_DIR / "debug_pcd" / "rewind" / "rewind_checkpoint_db.npz"))
    parser.add_argument("--out_json", default=str(THIS_DIR / "debug_pcd" / "rewind" / "rewind_checkpoint_db.json"))
    parser.add_argument("--config_name", default="robot_dp3")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--ckpt_setting", required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_num", type=int, required=True)
    parser.add_argument("--use_rgb", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=3)
    parser.add_argument("--kde_bandwidth", type=float, default=None, help="Gaussian KDE bandwidth in PCA space. Default: median within-slot distance.")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    model = load_model(args)

    slot_names = []
    features = []
    recovery_actions = []
    source_episodes = []
    frame_indices = []

    for episode_name, slots in annotations["episodes"].items():
        episode_path = find_episode_path(args.data_dir, episode_name)
        with h5py.File(episode_path, "r") as root:
            total = int(root["/state/vector"].shape[0])
            for slot_name, frame_idx in iter_slot_frames(slots):
                frame_idx = int(frame_idx)
                if frame_idx < 0 or frame_idx >= total:
                    raise IndexError(f"{episode_name}:{slot_name} frame {frame_idx} out of range 0..{total - 1}")
                obs, recovery_action20 = read_frame_obs(root, frame_idx)
                feature = extract_feature(model, obs)
                slot_names.append(str(slot_name))
                features.append(feature)
                recovery_actions.append(recovery_action20.astype(np.float32))
                source_episodes.append(str(episode_name))
                frame_indices.append(frame_idx)
                print(f"added {episode_name} {slot_name} frame={frame_idx}")

    if len(features) == 0:
        raise RuntimeError("No rewind checkpoint samples were collected.")
    pca_mean, pca_components, pca_features = fit_pca(np.asarray(features, dtype=np.float32), args.pca_dim)
    kde_bandwidth = estimate_bandwidth(pca_features, slot_names, args.kde_bandwidth)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        slot_names=np.asarray(slot_names),
        features=np.asarray(features, dtype=np.float32),
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_features=pca_features,
        kde_bandwidth=np.asarray(kde_bandwidth, dtype=np.float32),
        recovery_actions=np.asarray(recovery_actions, dtype=np.float32),
        source_episodes=np.asarray(source_episodes),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
    )
    metadata = {
        "num_checkpoints": len(slot_names),
        "slots": sorted(set(slot_names)),
        "pca_dim": int(pca_components.shape[0]),
        "kde_bandwidth": float(kde_bandwidth),
        "npz": os.path.abspath(args.out_npz),
        "entries": [
            {
                "slot_name": s,
                "episode": e,
                "frame_idx": int(f),
            }
            for s, e, f in zip(slot_names, source_episodes, frame_indices)
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out_npz}")
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
