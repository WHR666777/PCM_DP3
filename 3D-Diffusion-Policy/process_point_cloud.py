import h5py
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from semantic_extractor import SemanticPointExtractor  # 导入你的 DINOv2 类

def clean_single_episode(input_hdf5_path, output_hdf5_path, extractor, npy_save_dir=None, max_npy_frames=300):
    """处理单个 HDF5 文件的核心逻辑"""
    print(f"\n📂 正在处理: {os.path.basename(input_hdf5_path)}")
    
    with h5py.File(input_hdf5_path, 'r') as f_in, h5py.File(output_hdf5_path, 'w') as f_out:
        
        # 复制原始数据集的结构到新文件
        def copy_node(name, node):
            if isinstance(node, h5py.Dataset):
                if name != "pointcloud":  # 除了 pointcloud，其他全盘照抄
                    f_out.create_dataset(name, data=node[...])
        f_in.visititems(copy_node)
        
        # 获取需要处理的数据
        pcds = f_in['pointcloud'][...]               # (T, 1024, 6)
        rgbs = f_in['observation/head_camera/rgb'][...] # (T, byte_string)
        extrinsics = f_in['observation/head_camera/extrinsic_cv'][...] # (T, 3, 4)
        intrinsics = f_in['observation/head_camera/intrinsic_cv'][...] # (T, 3, 3)
        
        T = pcds.shape[0]
        target_num = pcds.shape[1] # 1024
        
        clean_pcds = np.zeros_like(pcds)
        
        print(f"🧹 开始逐帧清洗点云 (共 {T} 帧)...")
        for i in tqdm(range(T), leave=False):
            # 1. 解码 RGB 图像
            raw_bytes = rgbs[i]
            img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            
            # 2. 获取 DINOv2 掩膜 (0=背景/机械臂, 1=可乐, 2=篮子)
            mask = extractor.predict(img_rgb)
            target_mask_2d = (mask == 1) | (mask == 2)
            
            # 3. 提取当前帧的 1024 个 3D 点、内参、外参
            pts_3d = pcds[i, :, :3]  
            ext = extrinsics[i]      
            K = intrinsics[i]        
            
            # --- 核心：3D 投影回 2D ---
            pts_3d_homo = np.hstack((pts_3d, np.ones((target_num, 1)))) 
            pts_cam = (ext @ pts_3d_homo.T).T 
            pts_2d_homo = (K @ pts_cam.T).T 
            
            z_c = np.clip(pts_2d_homo[:, 2], a_min=1e-5, a_max=None)
            u = (pts_2d_homo[:, 0] / z_c).astype(int)
            v = (pts_2d_homo[:, 1] / z_c).astype(int)
            
            valid_projection = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts_cam[:, 2] > 0)
            
            # 4. 掩膜过滤
            final_valid_idx = []
            for idx in range(target_num):
                if valid_projection[idx]:
                    if target_mask_2d[v[idx], u[idx]]:
                        final_valid_idx.append(idx)
                        
            # 提取存活下来的点
            if len(final_valid_idx) == 0:
                clean_frame = np.zeros((target_num, 6), dtype=np.float32)
                # 过滤掉过多刷屏，只在控制台输出一次警告
                if i == 0 or i % 50 == 0:
                    print(f"\r⚠️ 警告：检测到无目标帧 (例如第 {i} 帧)", end="")
            else:
                survived_pts = pcds[i, final_valid_idx, :]
                
                # 5. 【有放回重采样】
                resample_indices = np.random.choice(len(survived_pts), target_num, replace=True)
                clean_frame = survived_pts[resample_indices]
                
            clean_pcds[i] = clean_frame
            
            # ==========================================
            # 新增：如果这是被选中的文件，保存前 300 帧为 npy
            # ==========================================
            if npy_save_dir is not None and i < max_npy_frames:
                npy_filename = os.path.join(npy_save_dir, f"frame_{i:03d}.npy")
                np.save(npy_filename, clean_frame)
                
        # 6. 保存新数据集
        f_out.create_dataset('pointcloud', data=clean_pcds)
        print(f"\n✅ 完成: {os.path.basename(output_hdf5_path)}")

def process_directory(input_dir, output_dir, head_weight_path, npy_vis_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(npy_vis_dir, exist_ok=True)
    
    # 查找输入目录下所有的 hdf5 文件
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    hdf5_files.sort() # 排序以保证一致性
    
    if not hdf5_files:
        print(f"❌ 目录 {input_dir} 中没有找到任何 .hdf5 文件！")
        return

    print(f"🚀 共找到 {len(hdf5_files)} 个数据片段。正在初始化 DINOv2 感知引擎...")
    # 只在最外层初始化一次提取器，避免重复加载权重，节省显存和时间
    extractor = SemanticPointExtractor(head_weight_path)
    
    for idx, input_path in enumerate(hdf5_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        # 逻辑：挑第一个文件 (idx == 0) 保存其前 300 帧作为可视化样本
        # 如果你想挑别的文件，改这里的 idx 判断即可
        current_npy_dir = npy_vis_dir if idx == 0 else None
        
        if current_npy_dir:
            print(f"\n👁️ [可视化开启] 将从 {filename} 提取前 300 帧点云保存到: {npy_vis_dir}")
            
        clean_single_episode(input_path, output_path, extractor, npy_save_dir=current_npy_dir)

    print(f"\n🎉 整个文件夹处理完毕！所有清洗后的 HDF5 文件已存入: {output_dir}")

if __name__ == "__main__":
    # 配置你的目录路径
    # 注意：这里改成了文件夹路径，而不是单个文件
    INPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/"
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/point_processed_data/" 
    HEAD_WEIGHT = "checkpoints/dinov2_linear_head.pth"
    
    # 用于存放抽帧检查的 NPY 文件夹
    NPY_VIS_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/vis_npy/"
    
    process_directory(INPUT_DIR, OUTPUT_DIR, HEAD_WEIGHT, NPY_VIS_DIR)