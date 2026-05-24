import h5py
import numpy as np
import os
import glob
from tqdm import tqdm # 用于显示进度条

def process_all_episodes(input_dir, output_dir, pos_threshold=1e-4, grip_threshold=0.05):
    """
    批量处理目录下的所有 hdf5 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 hdf5 文件路径
    search_pattern = os.path.join(input_dir, "*.hdf5")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"在 {input_dir} 中没有找到 .hdf5 文件！")
        return

    print(f"共找到 {len(file_list)} 个文件，开始批量清洗...")
    
    total_original_frames = 0
    total_filtered_frames = 0
    
    for input_path in tqdm(file_list, desc="Processing HDF5"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        with h5py.File(input_path, 'r') as f_in:
            num_frames = f_in['endpose/left_endpose'].shape[0]
            total_original_frames += num_frames
            
            left_eef = f_in['endpose/left_endpose'][:]
            right_eef = f_in['endpose/right_endpose'][:]
            left_grip = f_in['endpose/left_gripper'][:]
            right_grip = f_in['endpose/right_gripper'][:]
            
            valid_indices = []
            
            for i in range(num_frames):
                if i == 0 or i == num_frames - 1:
                    valid_indices.append(i)
                    continue
                    
                delta_left_pos = np.linalg.norm(left_eef[i, :3] - left_eef[i-1, :3])
                delta_right_pos = np.linalg.norm(right_eef[i, :3] - right_eef[i-1, :3])
                delta_left_grip = abs(left_grip[i] - left_grip[i-1])
                delta_right_grip = abs(right_grip[i] - right_grip[i-1])
                
                if (delta_left_pos > pos_threshold) or (delta_right_pos > pos_threshold) or \
                   (delta_left_grip > grip_threshold) or (delta_right_grip > grip_threshold):
                    valid_indices.append(i)
                    
            valid_indices = np.array(valid_indices)
            total_filtered_frames += len(valid_indices)
            
            with h5py.File(output_path, 'w') as f_out:
                def copy_filtered(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        f_out.create_dataset(name, data=obj[:][valid_indices])
                f_in.visititems(copy_filtered)
                
    print("\n========== 处理完成 ==========")
    print(f"总原始帧数: {total_original_frames}")
    print(f"总保留帧数: {total_filtered_frames}")
    print(f"共剔除了 {total_original_frames - total_filtered_frames} 个静止帧。")
    print(f"清洗后的数据已保存在: {output_dir}")

if __name__ == "__main__":
    # 配置输入和输出文件夹
    INPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data"
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data_filtered"
    
    process_all_episodes(INPUT_DIR, OUTPUT_DIR)