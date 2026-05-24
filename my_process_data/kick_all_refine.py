import h5py
import numpy as np
import os
import glob
from tqdm import tqdm

def process_basket_pause_only(input_dir, output_dir, pos_threshold=1e-4, grip_threshold=0.01, max_keep_idle=10):
    os.makedirs(output_dir, exist_ok=True)
    file_list = glob.glob(os.path.join(input_dir, "*.hdf5"))
    
    if not file_list:
        print(f"在 {input_dir} 中没有找到 .hdf5 文件！")
        return

    print(f"共找到 {len(file_list)} 个文件，开始精准压缩最后阶段静止帧...")
    
    total_original_frames = 0
    total_kept_frames = 0
    
    for input_path in tqdm(file_list, desc="Processing HDF5"):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        with h5py.File(input_path, 'r') as f_in:
            left_eef = f_in['endpose/left_endpose'][:]
            right_eef = f_in['endpose/right_endpose'][:]
            left_grip = f_in['endpose/left_gripper'][:]
            right_grip = f_in['endpose/right_gripper'][:]
            
            num_frames = left_eef.shape[0]
            total_original_frames += num_frames
            
            # 1. 标记每一帧是否在运动
            active_flags = np.zeros(num_frames, dtype=bool)
            for i in range(1, num_frames):
                dl_pos = np.linalg.norm(left_eef[i, :3] - left_eef[i-1, :3])
                dr_pos = np.linalg.norm(right_eef[i, :3] - right_eef[i-1, :3])
                dl_grip = abs(left_grip[i] - left_grip[i-1])
                dr_grip = abs(right_grip[i] - right_grip[i-1])
                
                if (dl_pos > pos_threshold) or (dr_pos > pos_threshold) or \
                   (dl_grip > grip_threshold) or (dr_grip > grip_threshold):
                    active_flags[i] = True

            # 2. 找到夹爪最后一次发生明显动作的索引 (这就是抓取篮子提手的时刻)
            grip_changes = np.where((abs(np.diff(left_grip)) > grip_threshold) | 
                                    (abs(np.diff(right_grip)) > grip_threshold))[0]
            
            # 如果没找到夹爪动作，默认保护前半段
            last_grip_idx = grip_changes[-1] if len(grip_changes) > 0 else num_frames // 2
            
            valid_indices = []
            idle_counter = 0
            
            # 3. 核心过滤逻辑：免检前半段，压缩后半段
            for i in range(num_frames):
                if i <= last_grip_idx:
                    # 【绝对保护区】：抓取篮子之前的所有动作（包括抓Can的停顿），100%保留
                    valid_indices.append(i)
                    idle_counter = 0
                else:
                    # 【精准压缩区】：抓取篮子之后的静止帧
                    if not active_flags[i]:
                        idle_counter += 1
                        # 超过 max_keep_idle (默认10帧) 的静止帧将被丢弃
                        if idle_counter <= max_keep_idle:
                            valid_indices.append(i)
                    else:
                        # 机械臂开始向上提篮子了，恢复保留
                        idle_counter = 0
                        valid_indices.append(i)
                        
            total_kept_frames += len(valid_indices)
            
            # 写入新文件
            with h5py.File(output_path, 'w') as f_out:
                def copy_filtered(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        f_out.create_dataset(name, data=obj[:][valid_indices])
                f_in.visititems(copy_filtered)
                
    print("\n========== 处理完成 ==========")
    print(f"总原始帧数: {total_original_frames}")
    print(f"总保留帧数: {total_kept_frames}")
    print(f"精准剔除了后半段的 {total_original_frames - total_kept_frames} 个冗余等待帧。")
    print(f"安全清洗后的数据已保存在: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data_original"
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data"
    
    # max_keep_idle=10 表示抓稳篮子后，强制保留 10 帧 (假设10Hz即1秒) 让物理引擎算摩擦力
    process_basket_pause_only(INPUT_DIR, OUTPUT_DIR, max_keep_idle=10)