import h5py
import cv2
import numpy as np
import os
from tqdm import tqdm

def export_h5_to_video(h5_path, output_path, cam_list=None, fps=20):
    """
    将 RoboTwin HDF5 数据导出为视频
    :param cam_list: 摄像头列表，例如 ['front_camera', 'head_camera']
    """
    if cam_list is None:
        cam_list = ['front_camera']
    
    with h5py.File(h5_path, 'r') as f:
        # 获取总帧数
        num_frames = f['endpose/left_endpose'].shape[0]
        
        writer = None
        
        for i in tqdm(range(num_frames), desc=f"正在生成: {os.path.basename(output_path)}"):
            combined_frames = []
            
            for cam in cam_list:
                rgb_key = f'observation/{cam}/rgb'
                if rgb_key not in f:
                    continue
                
                # 1. 读取字节流数据 (dtype 为 |S... 的数据)
                img_bytes = f[rgb_key][i]
                
                # 2. 解码为图像数组
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    combined_frames.append(frame)
            
            if not combined_frames:
                continue
                
            # 将多个视角横向拼接
            final_frame = np.hstack(combined_frames)
            
            # 初始化写入器
            if writer is None:
                h, w = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            writer.write(final_frame)
            
        if writer:
            writer.release()
            print(f"\n✅ 视频生成完毕: {output_path}")

if __name__ == "__main__":
    # 配置你的路径
    RAW_H5 = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data_original/episode2.hdf5"
    FILT_H5 = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/episode2.hdf5"
    
    # 建议查看 front_camera 和 head_camera 两个视角
    cameras = ['front_camera', 'head_camera']
    
    print("正在处理原始数据...")
    export_h5_to_video(RAW_H5, "visual_raw.mp4", cam_list=cameras)
    
    print("\n正在处理清洗后的数据...")
    export_h5_to_video(FILT_H5, "visual_filtered.mp4", cam_list=cameras)
    
    print("\n💡 提示：你可以使用浏览器的文件管理器下载这两个 mp4 文件进行对比。")
    print("清洗后的视频应该在抓取篮子后动作更连贯，没有长时间的定格。")