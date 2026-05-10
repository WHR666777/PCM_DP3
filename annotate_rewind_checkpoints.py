import argparse
import json
import os
import sys
from pathlib import Path

cv2 = None
h5py = None
np = None
o3d = None


DEFAULT_SLOTS = ["pre_grasp", "grasp_ready", "pre_place"]


def load_runtime_deps():
    global cv2, h5py, np, o3d
    import cv2 as _cv2
    import h5py as _h5py
    import numpy as _np
    cv2 = _cv2
    h5py = _h5py
    np = _np
    try:
        import open3d as _o3d
    except Exception:
        _o3d = None
    o3d = _o3d


def decode_rgb_image(image):
    if isinstance(image, (bytes, bytearray, np.bytes_)):
        arr = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
        if arr.size > 0 and np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def read_dataset(root, candidates):
    for path in candidates:
        if path in root:
            return root[path]
    return None


def get_frame_count(root):
    for path in [
        "/observation/head_camera/rgb",
        "/pointcloud",
        "/state/vector",
        "/joint_action/vector",
    ]:
        if path in root:
            return int(root[path].shape[0])
    raise KeyError("Cannot infer frame count from hdf5.")


def load_existing_annotations(path):
    if path is None or not os.path.isfile(path):
        return {"slot_names": DEFAULT_SLOTS, "episodes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(path, annotations):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


def append_slot_frame(episode_ann, slot, frame_idx):
    old_value = episode_ann.get(slot)
    if old_value is None:
        episode_ann[slot] = [int(frame_idx)]
        return
    if isinstance(old_value, list):
        values = [int(v) for v in old_value]
    else:
        values = [int(old_value)]
    if int(frame_idx) not in values:
        values.append(int(frame_idx))
    episode_ann[slot] = sorted(values)


def format_slot_marks(value):
    if value is None:
        return ""
    if isinstance(value, list):
        if len(value) <= 3:
            return ",".join(str(int(v)) for v in value)
        return f"{len(value)} samples,last={int(value[-1])}"
    return str(int(value))


def dbscan_keep_two(pcd, eps, min_points):
    if pcd is None or pcd.shape[0] == 0 or o3d is None:
        return pcd
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pcd
    counts = np.bincount(valid)
    top = np.argsort(counts)[-2:]
    return pcd[np.isin(labels, top)]


def draw_projection(pcd, title, size=(420, 300)):
    panel = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    if pcd is None or pcd.shape[0] == 0:
        cv2.putText(panel, "no points", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
        return panel
    pts = np.asarray(pcd[:, :3])
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return panel
    x_min, x_max = -0.65, 0.65
    y_min, y_max = -0.65, 0.65
    margin = 24
    px = margin + (pts[:, 0] - x_min) / (x_max - x_min) * (size[0] - 2 * margin)
    py = size[1] - margin - (pts[:, 1] - y_min) / (y_max - y_min) * (size[1] - 2 * margin)
    px = np.clip(px, margin, size[0] - margin - 1).astype(np.int32)
    py = np.clip(py, margin, size[1] - margin - 1).astype(np.int32)
    for x, y in zip(px, py):
        cv2.circle(panel, (int(x), int(y)), 2, (40, 130, 230), -1, lineType=cv2.LINE_AA)
    cv2.putText(panel, f"points={pts.shape[0]}", (12, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    return panel


def make_mask_overlay(rgb, mask):
    overlay = rgb.copy()
    if mask is None:
        return overlay
    mask = np.asarray(mask)
    color = np.zeros_like(overlay)
    color[mask == 1] = np.array([255, 60, 60], dtype=np.uint8)
    color[mask == 2] = np.array([60, 150, 255], dtype=np.uint8)
    target = (mask == 1) | (mask == 2)
    overlay[target] = (0.55 * overlay[target] + 0.45 * color[target]).astype(np.uint8)
    return overlay


def load_semantic_extractor(head_path):
    if not head_path:
        return None
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from semantic_extractor import SemanticPointExtractor
    except Exception:
        sys.path.append(str(Path(__file__).resolve().parent / "3D-Diffusion-Policy"))
        from semantic_extractor import SemanticPointExtractor
    return SemanticPointExtractor(head_weight_path=head_path)


def render_frame(root, frame_idx, extractor, eps, min_points, slots, episode_ann):
    rgb_ds = read_dataset(root, ["/observation/head_camera/rgb", "/observation/rgb/head_camera", "/rgb"])
    pcd_ds = read_dataset(root, ["/pointcloud"])
    rgb = decode_rgb_image(rgb_ds[frame_idx]) if rgb_ds is not None else None
    if rgb is None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = extractor.predict(rgb) if extractor is not None else None
    overlay = make_mask_overlay(rgb, mask)
    raw_pcd = np.asarray(pcd_ds[frame_idx]) if pcd_ds is not None else np.zeros((0, 6), dtype=np.float32)
    db_pcd = dbscan_keep_two(raw_pcd, eps=eps, min_points=min_points)

    rgb_panel = cv2.resize(rgb, (420, 300), interpolation=cv2.INTER_AREA)
    overlay_panel = cv2.resize(overlay, (420, 300), interpolation=cv2.INTER_AREA)
    raw_panel = draw_projection(raw_pcd, "Raw point cloud")
    db_panel = draw_projection(db_pcd, "DBSCAN top-2")
    top = np.concatenate([rgb_panel, overlay_panel], axis=1)
    bottom = np.concatenate([raw_panel, db_panel], axis=1)
    frame = np.concatenate([top, bottom], axis=0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 42), (20, 20, 20), -1)
    cv2.putText(frame, f"frame={frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    x = 160
    for i, slot in enumerate(slots):
        marked = episode_ann.get(slot)
        text = f"{i + 1}:{slot}"
        if marked is not None:
            text += f"={format_slot_marks(marked)}"
        cv2.putText(frame, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
        x += 190
    cv2.putText(frame, "a/d: prev/next  space: play  s: save  q: quit", (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to one episode hdf5.")
    parser.add_argument("--out", default="rewind_checkpoint_annotations.json")
    parser.add_argument("--head", default=None, help="Optional DINOv2 linear head path.")
    parser.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--min_points", type=int, default=150)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    load_runtime_deps()
    annotations = load_existing_annotations(args.out)
    annotations["slot_names"] = args.slots
    episode_key = os.path.basename(args.hdf5)
    annotations.setdefault("episodes", {}).setdefault(episode_key, {})
    episode_ann = annotations["episodes"][episode_key]
    extractor = load_semantic_extractor(args.head)

    with h5py.File(args.hdf5, "r") as root:
        total = get_frame_count(root)
        frame_idx = 0
        playing = False
        while True:
            panel = render_frame(root, frame_idx, extractor, args.eps, args.min_points, args.slots, episode_ann)
            cv2.imshow("Rewind checkpoint annotation", panel)
            key = cv2.waitKey(60 if playing else 0) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                playing = not playing
            elif key == ord("a"):
                frame_idx = max(0, frame_idx - 1)
            elif key == ord("d"):
                frame_idx = min(total - 1, frame_idx + 1)
            elif key == ord("j"):
                frame_idx = max(0, frame_idx - args.step)
            elif key == ord("l"):
                frame_idx = min(total - 1, frame_idx + args.step)
            elif key == ord("s"):
                save_annotations(args.out, annotations)
                print(f"saved annotations: {args.out}")
            elif ord("1") <= key <= ord("9"):
                slot_idx = key - ord("1")
                if slot_idx < len(args.slots):
                    append_slot_frame(episode_ann, args.slots[slot_idx], frame_idx)
                    print(f"{episode_key}: {args.slots[slot_idx]} += {frame_idx}")
            if playing:
                frame_idx = min(total - 1, frame_idx + 1)
    save_annotations(args.out, annotations)
    cv2.destroyAllWindows()
    print(f"saved annotations: {args.out}")


if __name__ == "__main__":
    main()
