import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# 导入你之前写好并调试通过的提取器类
from extractor import InteractionObjectExtractor

class DINOv2DataPreparer:
    # 增加 min_area 参数用于过滤遮挡；padding 调小到 3 实现“极限贴边”
    def __init__(self, data_dir, output_dir, padding=3, min_area=300):
        """
        :param data_dir: 存放 HDF5 专家数据的目录
        :param output_dir: 裁剪后图片的保存根目录
        :param padding: 裁剪框的外扩像素，不涂黑的情况下尽量小，减少背景干扰
        :param min_area: 有效像素面积阈值，低于此值的说明被严重遮挡，直接丢弃
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.padding = padding
        self.min_area = min_area
        
        # 创建输出分类文件夹
        self.can_dir = os.path.join(self.output_dir, "can")
        self.basket_dir = os.path.join(self.output_dir, "basket")
        os.makedirs(self.can_dir, exist_ok=True)
        os.makedirs(self.basket_dir, exist_ok=True)

    def get_bbox_from_mask(self, mask, img_shape):
        """从布尔掩码计算带 Padding 的 Bounding Box"""
        y_indices, x_indices = np.where(mask)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 增加 Padding 并防止越界
        h, w = img_shape[:2]
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_max = min(w, x_max + self.padding)
        y_max = min(h, y_max + self.padding)
        
        return x_min, y_min, x_max, y_max

    def decode_rgb(self, dataset, frame_idx):
        """读取并解码 RGB 图像"""
        raw_bytes = dataset['observation/head_camera/rgb'][frame_idx]
        rgb_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    def process_all_episodes(self):
        hdf5_files = glob.glob(os.path.join(self.data_dir, "*.hdf5"))
        print(f"🔍 找到 {len(hdf5_files)} 个 HDF5 文件，开始全量帧安全裁剪...")

        for file_path in tqdm(hdf5_files, desc="处理进度"):
            ep_name = os.path.splitext(os.path.basename(file_path))[0]
            
            try:
                extractor = InteractionObjectExtractor(file_path)
                grasp_info = extractor.detect_grasp_frames()
                total_frames = extractor.dataset['observation/head_camera/rgb'].shape[0]
                
                # === 1. 物理溯源：在第 0 帧获取最纯净的物体颜色 (真值 ID) ===
                # 获取可乐颜色
                can_frame = grasp_info['can_grasp']['frame']
                can_arm = grasp_info['can_grasp']['arm']
                u_can, v_can = extractor.project_3d_to_pixel(can_frame, can_arm)
                can_color = extractor.get_object_color(0, u_can, v_can)
                
                # 获取篮子颜色
                basket_frame = grasp_info['basket_grasp']['frame']
                basket_arm = grasp_info['basket_grasp']['arm']
                u_basket, v_basket = extractor.project_3d_to_pixel(basket_frame, basket_arm)
                basket_color = extractor.get_object_color(0, u_basket, v_basket)

                # === 2. 遍历当前 Episode 的【每一帧】 ===
                for frame_idx in range(total_frames):
                    # 统一读取当前帧的原图和分割图
                    rgb_img = self.decode_rgb(extractor.dataset, frame_idx)
                    seg_img = extractor.dataset['observation/head_camera/mesh_segmentation'][frame_idx]
                    
                    # ---------------- 提取 Can ----------------
                    mask_can = np.all(seg_img == can_color, axis=-1)
                    # 【面积过滤器】：像素数量大于阈值才认为有效，剔除严重遮挡
                    if np.sum(mask_can) >= self.min_area:
                        bbox_can = self.get_bbox_from_mask(mask_can, rgb_img.shape)
                        if bbox_can:
                            x1, y1, x2, y2 = bbox_can
                            # 【不涂黑】：直接极限贴边裁剪原图
                            cropped_can = rgb_img[y1:y2, x1:x2]
                            save_path = os.path.join(self.can_dir, f"{ep_name}_can_f{frame_idx:03d}.png")
                            cv2.imwrite(save_path, cv2.cvtColor(cropped_can, cv2.COLOR_RGB2BGR))

                    # ---------------- 提取 Basket ----------------
                    mask_basket = np.all(seg_img == basket_color, axis=-1)
                    # 篮子比较大，理论上面积阈值可以设得更高，这里统一使用 min_area
                    if np.sum(mask_basket) >= self.min_area:
                        bbox_basket = self.get_bbox_from_mask(mask_basket, rgb_img.shape)
                        if bbox_basket:
                            x1, y1, x2, y2 = bbox_basket
                            # 【不涂黑】：直接极限贴边裁剪原图
                            cropped_basket = rgb_img[y1:y2, x1:x2]
                            save_path = os.path.join(self.basket_dir, f"{ep_name}_basket_f{frame_idx:03d}.png")
                            cv2.imwrite(save_path, cv2.cvtColor(cropped_basket, cv2.COLOR_RGB2BGR))
                            
                extractor.dataset.close()
                
            except Exception as e:
                print(f"❌ 处理文件 {ep_name} 时发生错误: {str(e)}")
                continue

if __name__ == "__main__":
    # 配置你的输入输出路径
    DATA_DIR = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/"
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/data/dino_prototypes_data/"
    
    # 实例化并运行 (padding调为3实现极限贴边，min_area=300过滤残缺数据)
    preparer = DINOv2DataPreparer(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, padding=3, min_area=300)
    preparer.process_all_episodes()
    print("✅ 所有专家数据全量帧提取完毕！快去文件夹看看纯净度吧。")