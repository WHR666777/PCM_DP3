import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import open3d as o3d

from semantic_extractor import SemanticPointExtractor


PANEL_SIZE = (640, 360)  # width, height
POINT_PANEL_SIZE = (640, 360)
POINT_LIMITS_XY = (-0.65, 0.65, -0.65, 0.65)
POINT_LIMITS_XZ = (-0.65, 0.65, -0.25, 0.65)


def decode_rgb_image(image) -> Optional[np.ndarray]:
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


def get_matrix_frame(db: h5py.File, frame_idx: int, candidates: List[str]) -> np.ndarray:
    for path in candidates:
        if path in db:
            data = db[path][frame_idx]
            return np.asarray(data, dtype=np.float64).squeeze()
    raise KeyError(f"None of paths exist: {candidates}")


def normalize_intrinsic(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        mat = mat[:3, :3]
    elif mat.size == 9:
        mat = mat.reshape(3, 3)
    if mat.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsic shape: {mat.shape}")
    return mat


def normalize_extrinsic(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        mat = mat[:3, :]
    elif mat.size == 16:
        mat = mat.reshape(4, 4)[:3, :]
    elif mat.size == 12:
        mat = mat.reshape(3, 4)
    if mat.shape != (3, 4):
        raise ValueError(f"Unsupported extrinsic shape: {mat.shape}")
    return mat


def filter_pointcloud_by_same_frame_mask(
    pcd: np.ndarray, semantic_mask: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
) -> Tuple[np.ndarray, Dict[str, int]]:
    if pcd.shape[0] == 0:
        return pcd.copy(), {"raw_points": 0, "projected_points": 0, "kept_points": 0}

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
    valid_indices = np.where(valid_projection)[0]
    if valid_indices.size == 0:
        raise ValueError("no point projects into image")

    keep = np.zeros(pcd.shape[0], dtype=bool)
    keep[valid_indices] = target_mask[v[valid_indices], u[valid_indices]]

    filtered = pcd[keep].copy()
    stats = {
        "raw_points": int(pcd.shape[0]),
        "projected_points": int(valid_projection.sum()),
        "kept_points": int(keep.sum()),
    }
    return filtered, stats


def dbscan_two_object_result(
    pcd: np.ndarray, eps: float, min_points: int
) -> Tuple[np.ndarray, int]:
    if pcd.shape[0] == 0:
        return pcd.copy(), 0

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return pcd.copy(), 0

    counts = np.bincount(valid_labels)
    top_cluster_ids = np.argsort(counts)[-2:]
    keep = np.isin(labels, top_cluster_ids)
    cluster_count = len(top_cluster_ids)
    return pcd[keep].copy(), cluster_count


def normalize_point_colors(pcd: np.ndarray) -> np.ndarray:
    if pcd.shape[1] < 6:
        return np.tile(np.array([[30, 130, 250]], dtype=np.uint8), (pcd.shape[0], 1))
    colors = pcd[:, 3:6]
    if colors.size == 0:
        return np.tile(np.array([[30, 130, 250]], dtype=np.uint8), (pcd.shape[0], 1))
    colors = colors.astype(np.float64, copy=False)
    if np.nanmax(colors) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def draw_point_projection(
    canvas: np.ndarray,
    pcd: np.ndarray,
    rect: Tuple[int, int, int, int],
    axes: Tuple[int, int],
    limits: Tuple[float, float, float, float],
    title: str,
) -> None:
    x0, y0, width, height = rect
    margin = 16
    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (210, 210, 210), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

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


def make_point_panel(
    pcd: np.ndarray,
    title_prefix: str,
    bottom_text: str = "",
    output_size: Tuple[int, int] = POINT_PANEL_SIZE,
) -> np.ndarray:
    panel = np.full((output_size[1], output_size[0], 3), 245, dtype=np.uint8)
    pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 3:
        cv2.putText(panel, "Bad point cloud", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        return panel

    top_rect = (0, 0, output_size[0], output_size[1] // 2)
    bottom_rect = (0, output_size[1] // 2, output_size[0], output_size[1] // 2)
    draw_point_projection(panel, pcd, top_rect, axes=(0, 1), limits=POINT_LIMITS_XY, title=f"{title_prefix}: XY")
    draw_point_projection(panel, pcd, bottom_rect, axes=(0, 2), limits=POINT_LIMITS_XZ, title=f"{title_prefix}: XZ")
    cv2.putText(panel, f"points={pcd.shape[0]}", (14, output_size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    if bottom_text:
        cv2.putText(panel, bottom_text, (14, output_size[1] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return panel


def make_rgb_panel(rgb: Optional[np.ndarray], title: str) -> np.ndarray:
    if rgb is None:
        panel = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(panel, f"{title}: no image", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return panel
    panel = cv2.resize(rgb, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    cv2.putText(panel, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return panel


def make_mask_heatmap_panel(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    mapped = (mask_u8 * 100).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    panel = cv2.resize(heat_rgb, PANEL_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel, "Mask heatmap", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(panel, "class 0=bg, 1=can, 2=basket", (16, PANEL_SIZE[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def make_overlay_panel(rgb: Optional[np.ndarray], mask: np.ndarray) -> np.ndarray:
    if rgb is None:
        panel = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(panel, "Overlay: no image", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return panel
    overlay = rgb.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[mask == 1] = np.array([255, 70, 70], dtype=np.uint8)
    color_layer[mask == 2] = np.array([70, 150, 255], dtype=np.uint8)
    target = (mask == 1) | (mask == 2)
    overlay[target] = (0.55 * overlay[target] + 0.45 * color_layer[target]).astype(np.uint8)
    panel = cv2.resize(overlay, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    cv2.putText(panel, "RGB + mask overlay", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return panel


def build_frame_panel(
    frame_idx: int,
    rgb: Optional[np.ndarray],
    mask: np.ndarray,
    pcd_raw: np.ndarray,
    pcd_masked: np.ndarray,
    pcd_dbscan: np.ndarray,
    projection_stats: Dict[str, int],
    cluster_count: int,
    fallback: bool,
    fallback_note: str,
) -> np.ndarray:
    rgb_panel = make_rgb_panel(rgb, "Raw RGB")
    heatmap_panel = make_mask_heatmap_panel(mask)
    overlay_panel = make_overlay_panel(rgb, mask)
    raw_panel = make_point_panel(pcd_raw, "Raw PCD")

    masked_text = (
        f"raw={projection_stats.get('raw_points', 0)} "
        f"proj={projection_stats.get('projected_points', 0)} "
        f"kept={projection_stats.get('kept_points', 0)}"
    )
    masked_panel = make_point_panel(pcd_masked, "Mask-filtered PCD", masked_text)

    db_text = f"clusters={cluster_count}"
    if fallback:
        db_text = f"{db_text} fallback=1"
    db_panel = make_point_panel(pcd_dbscan, "DBSCAN PCD", db_text)

    top = np.concatenate([rgb_panel, heatmap_panel, overlay_panel], axis=1)
    bottom = np.concatenate([raw_panel, masked_panel, db_panel], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    cv2.putText(canvas, f"frame={frame_idx:04d}", (16, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if fallback and fallback_note:
        cv2.putText(canvas, f"fallback: {fallback_note}", (260, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def make_contact_sheet(
    sampled_rgb_panels: List[np.ndarray], out_path: str, thumb_size: Tuple[int, int] = (480, 180), n_cols: int = 5
) -> None:
    if not sampled_rgb_panels:
        return
    thumbs = [cv2.resize(img, thumb_size, interpolation=cv2.INTER_AREA) for img in sampled_rgb_panels]
    n = len(thumbs)
    n_rows = int(np.ceil(n / n_cols))
    cell_w, cell_h = thumb_size
    sheet = np.full((n_rows * cell_h, n_cols * cell_w, 3), 18, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // n_cols
        c = i % n_cols
        y0 = r * cell_h
        x0 = c * cell_w
        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = thumb
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def process_episode(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "stats.csv")
    summary_path = os.path.join(args.out_dir, "summary_contact_sheet.png")

    extractor = SemanticPointExtractor(head_weight_path=args.head)
    sampled_panels: List[np.ndarray] = []
    csv_rows: List[Dict[str, object]] = []

    with h5py.File(args.hdf5, "r") as db:
        if "pointcloud" not in db or "observation/head_camera/rgb" not in db:
            raise KeyError("hdf5 must contain /pointcloud and /observation/head_camera/rgb")
        pointcloud_ds = db["pointcloud"]
        rgb_ds = db["observation/head_camera/rgb"]
        total_frames = min(args.max_frames, pointcloud_ds.shape[0], rgb_ds.shape[0])

        print(f"processing {total_frames} frame(s) from: {args.hdf5}")
        for i in range(total_frames):
            pcd_raw = np.asarray(pointcloud_ds[i])
            rgb = decode_rgb_image(rgb_ds[i])
            if rgb is None:
                mask = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0]), dtype=np.uint8)
                pcd_masked = pcd_raw.copy()
                projection_stats = {"raw_points": int(pcd_raw.shape[0]), "projected_points": 0, "kept_points": int(pcd_raw.shape[0])}
                fallback = True
                fallback_note = "rgb decode failed"
            else:
                mask = extractor.predict(rgb)
                fallback = False
                fallback_note = ""
                try:
                    intrinsic = normalize_intrinsic(
                        get_matrix_frame(
                            db,
                            i,
                            ["observation/head_camera/intrinsic_cv", "observation/head_camera/intrinsic"],
                        )
                    )
                    extrinsic = normalize_extrinsic(
                        get_matrix_frame(
                            db,
                            i,
                            ["observation/head_camera/extrinsic_cv", "observation/head_camera/extrinsic"],
                        )
                    )
                    pcd_masked, projection_stats = filter_pointcloud_by_same_frame_mask(
                        pcd_raw, mask, intrinsic, extrinsic
                    )
                except Exception as e:
                    pcd_masked = pcd_raw.copy()
                    projection_stats = {
                        "raw_points": int(pcd_raw.shape[0]),
                        "projected_points": 0,
                        "kept_points": int(pcd_raw.shape[0]),
                    }
                    fallback = True
                    fallback_note = str(e)

            pcd_dbscan, cluster_count = dbscan_two_object_result(
                pcd_masked, eps=args.eps, min_points=args.min_points
            )
            frame_panel = build_frame_panel(
                frame_idx=i,
                rgb=rgb,
                mask=mask,
                pcd_raw=pcd_raw,
                pcd_masked=pcd_masked,
                pcd_dbscan=pcd_dbscan,
                projection_stats=projection_stats,
                cluster_count=cluster_count,
                fallback=fallback,
                fallback_note=fallback_note,
            )
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_panel, cv2.COLOR_RGB2BGR))

            if i % args.sample_stride == 0:
                sampled_panels.append(frame_panel)

            csv_rows.append(
                {
                    "frame_idx": i,
                    "raw_points": int(pcd_raw.shape[0]),
                    "projected_points": int(projection_stats.get("projected_points", 0)),
                    "mask_kept_points": int(projection_stats.get("kept_points", 0)),
                    "dbscan_kept_points": int(pcd_dbscan.shape[0]),
                    "cluster_count": int(cluster_count),
                    "fallback": int(fallback),
                    "note": fallback_note,
                }
            )
            if (i + 1) % 20 == 0 or i == total_frames - 1:
                print(f"  done {i + 1}/{total_frames}")

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "raw_points",
                "projected_points",
                "mask_kept_points",
                "dbscan_kept_points",
                "cluster_count",
                "fallback",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    make_contact_sheet(sampled_panels, summary_path)
    print(f"saved frames: {frames_dir}")
    print(f"saved stats : {stats_path}")
    print(f"saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DINOv2 mask + DBSCAN on hdf5 episodes.")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to episode .hdf5 file.")
    parser.add_argument("--head", type=str, required=True, help="Path to dinov2 linear head weights.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for images and stats.")
    parser.add_argument("--max_frames", type=int, default=300, help="Max number of frames to process.")
    parser.add_argument("--eps", type=float, default=0.04, help="DBSCAN eps.")
    parser.add_argument("--min_points", type=int, default=150, help="DBSCAN min_points.")
    parser.add_argument("--sample_stride", type=int, default=10, help="Sampling stride for contact sheet.")
    return parser.parse_args()


if __name__ == "__main__":
    process_episode(parse_args())
