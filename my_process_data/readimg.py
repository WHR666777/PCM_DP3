import h5py
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def extract_head_camera_rgb(hdf5_path, output_dir, max_frames=None):
    """
    从 HDF5 中提取 head_camera 的 RGB 图像并保存为 PNG 文件
    :param hdf5_path: HDF5 文件路径
    :param output_dir: 输出目录
    :param max_frames: 最大提取帧数，None 表示全部提取
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as dataset:
        # 检查是否存在 head_camera 数据
        rgb_key = 'observation/head_camera/rgb'
        if rgb_key not in dataset:
            raise KeyError(f"数据集中未找到 {rgb_key}，请检查 HDF5 结构")

        num_frames = dataset[rgb_key].shape[0]
        print(f"共有 {num_frames} 帧 RGB 图像")

        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        for i in tqdm(range(num_frames), desc="提取图像"):
            raw_bytes = dataset[rgb_key][i]
            # 解码 JPEG 字节流
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"警告：第 {i} 帧解码失败，跳过")
                continue

            # 默认 BGR 转 RGB 保存（opencv 写图片时会按 BGR 保存，所以保留 BGR 即可）
            # 如果想保存为 RGB 色彩顺序，可注释下行，或使用 cv2.COLOR_BGR2RGB 转换后保存
            save_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            cv2.imwrite(save_path, img)

    print(f"✅ 完成！图像已保存至 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 RoboTwin HDF5 数据集中提取 head_camera 的 RGB 图像")
    parser.add_argument("hdf5_path", type=str, help="HDF5 文件路径")
    parser.add_argument("--output_dir", "-o", type=str, default="./extracted_images",
                        help="输出目录 (默认: ./extracted_images)")
    parser.add_argument("--max_frames", "-n", type=int, default=None,
                        help="最大提取帧数 (默认提取全部)")
    args = parser.parse_args()

    extract_head_camera_rgb(args.hdf5_path, args.output_dir, args.max_frames)