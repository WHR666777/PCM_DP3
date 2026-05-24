import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def filter_single_episode(input_path, output_path, pos_threshold=1e-4, grip_threshold=0.05):
    """
    处理单个 HDF5 文件并剔除静止帧
    """
    with h5py.File(input_path, 'r') as f_in:
        num_frames = f_in['endpose/left_endpose'].shape[0]
        
        # 提取用于判断的末端和夹爪数据
        left_eef = f_in['endpose/left_endpose'][:]   # (N, 7) 位置+四元数
        right_eef = f_in['endpose/right_endpose'][:] # (N, 7)
        left_grip = f_in['endpose/left_gripper'][:]  # (N,)
        right_grip = f_in['endpose/right_gripper'][:] # (N,)
        
        valid_indices = []
        
        for i in range(num_frames):
            if i == 0 or i == num_frames - 1:
                valid_indices.append(i) # 始终保留首尾帧
                continue
                
            # 计算当前帧与上一帧的变化量 (只计算前3维 XYZ 的位置变化)
            delta_left_pos = np.linalg.norm(left_eef[i, :3] - left_eef[i-1, :3])
            delta_right_pos = np.linalg.norm(right_eef[i, :3] - right_eef[i-1, :3])
            
            # 计算夹爪指令的变化
            delta_left_grip = abs(left_grip[i] - left_grip[i-1])
            delta_right_grip = abs(right_grip[i] - right_grip[i-1])
            
            # 判断逻辑：如果左臂或右臂有明显移动，或者任一夹爪状态发生变化，则保留该帧
            is_moving = (delta_left_pos > pos_threshold) or (delta_right_pos > pos_threshold)
            is_gripping = (delta_left_grip > grip_threshold) or (delta_right_grip > grip_threshold)
            
            if is_moving or is_gripping:
                valid_indices.append(i)
                
        valid_indices = np.array(valid_indices)
        print(f"[{os.path.basename(input_path)}] 原始帧数: {num_frames} -> 过滤后帧数: {len(valid_indices)}")
        
        # 将有效数据写入新文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f_out:
            def copy_filtered(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # 针对时间维度（第0维）进行切片提取
                    f_out.create_dataset(name, data=obj[:][valid_indices])
            
            f_in.visititems(copy_filtered)
            
    return valid_indices

def plot_comparison(orig_path, filt_path):
    """
    绘制右臂（或左臂）Z轴高度的变化曲线，直观检查静止帧是否被删掉
    """
    with h5py.File(orig_path, 'r') as f_orig, h5py.File(filt_path, 'r') as f_filt:
        # 这里以右臂的 Z 轴（索引为2）为例，你可以根据实际抓篮子的是哪只手来修改
        orig_z = f_orig['endpose/right_endpose'][:, 2] 
        filt_z = f_filt['endpose/right_endpose'][:, 2]

        plt.figure(figsize=(12, 5))
        plt.plot(orig_z, label=f'Original ({len(orig_z)} frames)', color='#1f77b4', linewidth=2)
        plt.plot(filt_z, label=f'Filtered ({len(filt_z)} frames)', color='#ff7f0e', linestyle='--', linewidth=2)
        
        plt.title('Right Arm Z-Axis Trajectory Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Z Position')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 替换为你实际的路径
    orig_file = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/episode0.hdf5"
    filt_file = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data_test/episode0_filtered.hdf5"
    
    # 1. 执行过滤
    filter_single_episode(orig_file, filt_file)
    
    # 2. 绘制对比图
    plot_comparison(orig_file, filt_file)