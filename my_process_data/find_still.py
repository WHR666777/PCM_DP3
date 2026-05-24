import h5py
import numpy as np
import os
import glob

def analyze_tail_idle_frames(input_dir, grip_threshold=0.01):
    search_pattern = os.path.join(input_dir, "*.hdf5")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"在 {input_dir} 中没有找到 .hdf5 文件！")
        return

    print(f"共找到 {len(file_list)} 个文件，开始分析尾部静止等待帧...\n")
    print("-" * 50)
    
    idle_counts = []
    
    for input_path in file_list:
        filename = os.path.basename(input_path)
        with h5py.File(input_path, 'r') as f_in:
            left_grip = f_in['endpose/left_gripper'][:]
            right_grip = f_in['endpose/right_gripper'][:]
            
            num_frames = left_grip.shape[0]
            
            # 计算夹爪变化量
            dl_grip = abs(np.diff(left_grip))
            dr_grip = abs(np.diff(right_grip))
            
            # 找到所有夹爪动作大于阈值的索引
            grip_changes = np.where((dl_grip > grip_threshold) | (dr_grip > grip_threshold))[0]
            
            if len(grip_changes) > 0:
                # 最后一个动作的索引
                last_grip_idx = grip_changes[-1]
                # 计算从最后一次动作到结尾，拖了多少帧
                tail_idle_frames = num_frames - 1 - last_grip_idx
            else:
                tail_idle_frames = 0
                print(f"⚠️ 警告: {filename} 中未检测到任何夹爪动作！")
                
            idle_counts.append(tail_idle_frames)
            
            # 假设控制频率是 10Hz (每帧0.1秒)，换算成秒数更直观
            idle_time_sec = tail_idle_frames * 0.1 
            print(f"文件 {filename:<20} | 总帧数: {num_frames:<4} | 尾部静止: {tail_idle_frames:<4} 帧 (约 {idle_time_sec:.1f} 秒)")

    # 打印全局统计信息
    print("-" * 50)
    print("【全局统计结果】")
    print(f"平均尾部等待: {np.mean(idle_counts):.1f} 帧 (约 {np.mean(idle_counts)*0.1:.1f} 秒)")
    print(f"最长尾部等待: {np.max(idle_counts)} 帧 (约 {np.max(idle_counts)*0.1:.1f} 秒)")
    print(f"最短尾部等待: {np.min(idle_counts)} 帧 (约 {np.min(idle_counts)*0.1:.1f} 秒)")

if __name__ == "__main__":
    # 指向你原始的（未被破坏的）数据目录
    INPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data_original"
    
    analyze_tail_idle_frames(INPUT_DIR)