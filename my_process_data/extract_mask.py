import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# 导入你之前的提取器类
from extractor import InteractionObjectExtractor

class SegmentationDatasetBuilder:
    def __init__(self, data_dir, output_dir, sample_rate=5):
        """
        :param sample_rate: 抽样率，每隔多少帧提取一次（防止全量提取导致高度冗余和占用过大）
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        
        # 建立标准分割数据集目录结构
        self.img_dir = os.path.join(self.output_dir, "images")
        self.mask_dir = os.path.join(self.output_dir, "masks")
        # 建立可视化目录 (方便人类肉眼验证，因为 0,1,2 的 mask 图看上是全黑的)
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

    def process_all_episodes(self):
        hdf5_files = glob.glob(os.path.join(self.data_dir, "*.hdf5"))
        print(f"🔍 找到 {len(hdf5_files)} 个 HDF5 文件，开始构建全局分割数据集...")

        for file_path in tqdm(hdf5_files, desc="处理进度"):
            ep_name = os.path.splitext(os.path.basename(file_path))[0]
            
            try:
                extractor = InteractionObjectExtractor(file_path)
                grasp_info = extractor.detect_grasp_frames()
                
                # ================= 1. 物理溯源获取真值 ID (第 0 帧) =================
                can_frame = grasp_info['can_grasp']['frame']
                can_arm = grasp_info['can_grasp']['arm']
                u_can, v_can = extractor.project_3d_to_pixel(can_frame, can_arm)
                can_color = extractor.get_object_color(0, u_can, v_can)
                
                basket_frame = grasp_info['basket_grasp']['frame']
                basket_arm = grasp_info['basket_grasp']['arm']
                u_basket, v_basket = extractor.project_3d_to_pixel(basket_frame, basket_arm)
                basket_color = extractor.get_object_color(0, u_basket, v_basket)

                # ================= 2. 遍历序列进行抽样提取 =================
                total_frames = extractor.dataset['observation/head_camera/rgb'].shape[0]
                
                # 使用 sample_rate 进行跳帧提取
                for frame_idx in range(0, total_frames, self.sample_rate):
                    img_filename = f"{ep_name}_f{frame_idx:04d}.png"
                    
                    # --- A. 读取并保存全景 RGB (输入) ---
                    raw_bytes = extractor.dataset['observation/head_camera/rgb'][frame_idx]
                    img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    cv2.imwrite(os.path.join(self.img_dir, img_filename), img_bgr)
                    
                    # --- B. 构建严格的 0-1-2 单通道全景 Mask (标签) ---
                    seg_img = extractor.dataset['observation/head_camera/mesh_segmentation'][frame_idx]
                    h, w, _ = seg_img.shape
                    
                    # 初始化全 0 背景
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 匹配可乐，赋值为 1
                    mask[np.all(seg_img == can_color, axis=-1)] = 1
                    # 匹配篮子，赋值为 2
                    mask[np.all(seg_img == basket_color, axis=-1)] = 2
                    
                    # 保存 Mask，必须是不带任何压缩失真的 PNG
                    cv2.imwrite(os.path.join(self.mask_dir, img_filename), mask)
                    
                    # --- C. (可选辅助) 生成肉眼可视化的校验图 ---
                    # 背景全黑，可乐红色，篮子绿色
                    viz_img = np.zeros((h, w, 3), dtype=np.uint8)
                    viz_img[mask == 1] = [0, 0, 255] # BGR 红色
                    viz_img[mask == 2] = [0, 255, 0] # BGR 绿色
                    
                    # 把掩码半透明叠加到原图上，方便对齐检查
                    viz_blend = cv2.addWeighted(img_bgr, 0.5, viz_img, 0.5, 0)
                    cv2.imwrite(os.path.join(self.viz_dir, img_filename), viz_blend)

                extractor.dataset.close()
                
            except Exception as e:
                print(f"❌ 处理文件 {ep_name} 时发生错误: {str(e)}")
                continue

if __name__ == "__main__":
    # 配置路径
    DATA_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/demo_randomized/data/"
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/data/dino_linear_probe_data/"
    
    # 抽样率: 5 表示每 5 帧抽 1 帧。如果数据量少，可以设为 1 (全量提取)
    builder = SegmentationDatasetBuilder(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, sample_rate=5)
    builder.process_all_episodes()
    print("✅ 全局分割数据集提取完毕！请检查 visualizations 目录确认对齐情况。")