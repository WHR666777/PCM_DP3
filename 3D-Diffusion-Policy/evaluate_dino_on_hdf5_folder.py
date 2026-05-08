import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from semantic_extractor import SemanticPointExtractor


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


def collect_hdf5_files(hdf5_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    files = sorted(hdf5_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def load_gt_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def update_confusion(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> None:
    valid = (y_true >= 0) & (y_true < n_cls) & (y_pred >= 0) & (y_pred < n_cls)
    if not np.any(valid):
        return
    idx = y_true[valid] * n_cls + y_pred[valid]
    hist = np.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    conf += hist


def metrics_from_confusion(conf: np.ndarray, classes: Tuple[int, ...] = (1, 2)) -> Dict[str, float]:
    eps = 1e-9
    total = conf.sum()
    pixel_acc = float(np.trace(conf) / max(total, 1))

    out: Dict[str, float] = {"pixel_acc_all": pixel_acc}
    ious = []
    dices = []
    recalls = []
    precisions = []
    for c in classes:
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - tp)
        fn = float(conf[c, :].sum() - tp)
        iou = tp / (tp + fp + fn + eps)
        dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        out[f"class{c}_iou"] = iou
        out[f"class{c}_dice"] = dice
        out[f"class{c}_recall"] = recall
        out[f"class{c}_precision"] = precision
        ious.append(iou)
        dices.append(dice)
        recalls.append(recall)
        precisions.append(precision)

    out["miou_target"] = float(np.mean(ious)) if ious else 0.0
    out["mdice_target"] = float(np.mean(dices)) if dices else 0.0
    out["mrecall_target"] = float(np.mean(recalls)) if recalls else 0.0
    out["mprecision_target"] = float(np.mean(precisions)) if precisions else 0.0
    return out


def evaluate_episode(
    hdf5_path: Path,
    mask_dir: Path,
    extractor: SemanticPointExtractor,
    rgb_key: str,
    max_frames: int,
) -> Dict[str, object]:
    episode_name = hdf5_path.stem
    conf = np.zeros((3, 3), dtype=np.int64)
    valid_frames = 0
    missing_gt = 0
    bad_rgb = 0

    with h5py.File(hdf5_path, "r") as f:
        if rgb_key not in f:
            raise KeyError(f"{hdf5_path}: missing dataset {rgb_key}")
        rgb_ds = f[rgb_key]
        n_frames = rgb_ds.shape[0]
        if max_frames > 0:
            n_frames = min(n_frames, max_frames)

        for frame_idx in range(n_frames):
            gt_path = mask_dir / f"{episode_name}_f{frame_idx:04d}.png"
            gt_mask = load_gt_mask(gt_path)
            if gt_mask is None:
                missing_gt += 1
                continue

            rgb = decode_rgb_image(rgb_ds[frame_idx])
            if rgb is None:
                bad_rgb += 1
                continue

            pred_mask = extractor.predict(rgb)
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            pred_mask = pred_mask.astype(np.uint8)
            gt_mask = gt_mask.astype(np.uint8)
            update_confusion(conf, gt_mask.reshape(-1), pred_mask.reshape(-1), n_cls=3)
            valid_frames += 1

    metrics = metrics_from_confusion(conf, classes=(1, 2))
    result: Dict[str, object] = {
        "episode": episode_name,
        "file": str(hdf5_path),
        "total_pixels": int(conf.sum()),
        "valid_frames": int(valid_frames),
        "missing_gt_frames": int(missing_gt),
        "bad_rgb_frames": int(bad_rgb),
        "confusion": conf.tolist(),
    }
    result.update(metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINO masks on all hdf5 files with GT mask folder.")
    parser.add_argument("--hdf5_dir", type=str, required=True, help="Folder containing hdf5 files.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Folder containing GT masks (*.png).")
    parser.add_argument("--head", type=str, required=True, help="Path to dinov2_linear_head.pth.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for metrics.")
    parser.add_argument("--rgb_key", type=str, default="observation/head_camera/rgb", help="RGB dataset key in hdf5.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames, otherwise limit per episode.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search hdf5 under hdf5_dir.")
    args = parser.parse_args()

    hdf5_dir = Path(args.hdf5_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_hdf5_files(hdf5_dir, recursive=args.recursive)
    if len(files) == 0:
        raise FileNotFoundError(f"No hdf5 found under {hdf5_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir does not exist: {mask_dir}")

    print(f"found {len(files)} hdf5 files")
    extractor = SemanticPointExtractor(head_weight_path=args.head)

    per_episode_rows: List[Dict[str, object]] = []
    conf_global = np.zeros((3, 3), dtype=np.int64)
    total_valid = 0
    total_missing = 0
    total_bad_rgb = 0

    for idx, hdf5_path in enumerate(files):
        print(f"[{idx + 1}/{len(files)}] evaluating {hdf5_path.name}")
        row = evaluate_episode(
            hdf5_path=hdf5_path,
            mask_dir=mask_dir,
            extractor=extractor,
            rgb_key=args.rgb_key,
            max_frames=args.max_frames,
        )
        per_episode_rows.append(row)
        conf_global += np.asarray(row["confusion"], dtype=np.int64)
        total_valid += int(row["valid_frames"])
        total_missing += int(row["missing_gt_frames"])
        total_bad_rgb += int(row["bad_rgb_frames"])

    global_metrics = metrics_from_confusion(conf_global, classes=(1, 2))
    summary = {
        "n_files": len(files),
        "total_valid_frames": total_valid,
        "total_missing_gt_frames": total_missing,
        "total_bad_rgb_frames": total_bad_rgb,
        "confusion_global": conf_global.tolist(),
        **global_metrics,
    }

    csv_path = out_dir / "episode_metrics.csv"
    json_path = out_dir / "summary_metrics.json"

    fields = [
        "episode",
        "file",
        "valid_frames",
        "missing_gt_frames",
        "bad_rgb_frames",
        "pixel_acc_all",
        "class1_iou",
        "class2_iou",
        "miou_target",
        "class1_dice",
        "class2_dice",
        "mdice_target",
        "class1_recall",
        "class2_recall",
        "mrecall_target",
        "class1_precision",
        "class2_precision",
        "mprecision_target",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_episode_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("========== Summary ==========")
    print(f"mIoU(target 1/2): {summary['miou_target']:.4f}")
    print(f"mDice(target 1/2): {summary['mdice_target']:.4f}")
    print(f"Pixel Acc(all): {summary['pixel_acc_all']:.4f}")
    print(f"valid frames: {summary['total_valid_frames']}")
    print(f"missing gt frames: {summary['total_missing_gt_frames']}")
    print(f"bad rgb frames: {summary['total_bad_rgb_frames']}")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
