import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class InteractionObjectExtractor:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.dataset = h5py.File(hdf5_path, 'r')
        
    def detect_grasp_frames(self):
        """
        通过计算夹爪状态的梯度，找出左右手闭合的帧索引。
        注意：根据夹爪硬件，闭合可能是数值变大或变小。
        这里使用数值绝对变化最剧烈的地方作为抓取发生点。
        """
        left_gripper = self.dataset['endpose/left_gripper'][:]
        right_gripper = self.dataset['endpose/right_gripper'][:]
        
        # 计算一阶差分（变化率）
        left_diff = np.abs(np.diff(left_gripper))
        right_diff = np.abs(np.diff(right_gripper))
        
        # 寻找变化率最大的帧，即抓取瞬间 (向后偏移几帧确保已经抓稳)
        left_grasp_frame = np.argmax(left_diff) + 5 
        right_grasp_frame = np.argmax(right_diff) + 5
        
        # 你的任务逻辑：先抓的一定是 Can，后抓的一定是 Basket
        grasps = [
            {'arm': 'left', 'frame': left_grasp_frame},
            {'arm': 'right', 'frame': right_grasp_frame}
        ]
        # 按发生时间排序
        grasps.sort(key=lambda x: x['frame'])
        
        return {
            'can_grasp': grasps[0],      # 第一个闭合的
            'basket_grasp': grasps[1]    # 第二个闭合的
        }

    def project_3d_to_pixel(self, frame_idx, arm_name):
        """将末端 3D 坐标投影到 head_camera 的 2D 像素系"""
        # 1. 获取 3D 坐标 (x, y, z)
        endpose_key = f'endpose/{arm_name}_endpose'
        pos_3d = self.dataset[endpose_key][frame_idx, :3].copy() # 加 .copy() 防止修改原数据
        
        # 假设 RoboTwin 的世界坐标系 Z 轴朝上，往下偏移 5cm (0.05m)
        pos_3d[2] -= 0.15 
        
        pos_3d_homo = np.append(pos_3d, 1.0) # 齐次坐标 [x, y, z, 1]
        
        # 2. 获取相机内外参
        extrinsic = self.dataset['observation/head_camera/extrinsic_cv'][frame_idx] # (3, 4)
        intrinsic = self.dataset['observation/head_camera/intrinsic_cv'][frame_idx] # (3, 3)
        
        # 3. 投影计算
        pos_cam = extrinsic @ pos_3d_homo           # 世界系 -> 相机系
        pos_pixel_homo = intrinsic @ pos_cam        # 相机系 -> 像素系
        
        u = int(pos_pixel_homo[0] / pos_pixel_homo[2])
        v = int(pos_pixel_homo[1] / pos_pixel_homo[2])
        
        v_offset = 0
        return u, v + v_offset


    def get_object_color(self, frame_idx, u, v, window_size=5):
        """在投影点周围取窗口，找出占比最大的颜色作为物体颜色"""
        seg_img = self.dataset['observation/head_camera/mesh_segmentation'][frame_idx]
        h, w, _ = seg_img.shape
        
        # 定义窗口边界，防止越界
        half_w = window_size // 2
        u_min, u_max = max(0, u - half_w), min(w, u + half_w + 1)
        v_min, v_max = max(0, v - half_w), min(h, v + half_w + 1)
        
        # 提取窗口内的所有颜色像素
        window_pixels = seg_img[v_min:v_max, u_min:u_max].reshape(-1, 3)
        
        # 将颜色转为 tuple 以便哈希统计
        colors = [tuple(color) for color in window_pixels]
        most_common_color = Counter(colors).most_common(1)[0][0]
        
        return np.array(most_common_color)

    def visualize_extraction(self):
        """运行提取逻辑并生成可视化验证图"""
        # 1. 解析抓取动作
        grasp_info = self.detect_grasp_frames()
        print(f"✅ 动作解析完成：")
        print(f"   -> [Can] 抓取臂: {grasp_info['can_grasp']['arm']}, 发生帧: {grasp_info['can_grasp']['frame']}")
        print(f"   -> [Basket] 抓取臂: {grasp_info['basket_grasp']['arm']}, 发生帧: {grasp_info['basket_grasp']['frame']}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Target Object Extraction Verification', fontsize=16)
        
        # 遍历两个目标物体
        for i, (obj_name, grasp_data) in enumerate(grasp_info.items()):
            frame = grasp_data['frame']
            arm = grasp_data['arm']
            
            # 计算投影并获取颜色
            u, v = self.project_3d_to_pixel(frame, arm)
            
            # 2. 【核心修复】：时间倒流！去抓取前 15 帧的画面里提取颜色
            # (限制最小为 0，防止索引越界)
            # look_back_frame = max(0, frame - 60) 
            look_back_frame = 0
            target_color = self.get_object_color(look_back_frame, u, v)
            import cv2
            
            # 1. 读取原始字节流数据
            raw_bytes = self.dataset['observation/head_camera/rgb'][look_back_frame]
            
            # 2. 解码并转换通道 (BGR 转为 RGB)
            rgb_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            seg_img = self.dataset['observation/head_camera/mesh_segmentation'][look_back_frame]
            
            # 生成目标物体的布尔掩码
            # 对比图像中所有像素与 target_color 是否完全一致
            mask = np.all(seg_img == target_color, axis=-1)
            
            # 可视化 1: 原始图像 + 投影点
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].plot(u, v, 'r*', markersize=15) # 用红星标出抓取投影点
            axes[i, 0].set_title(f'{obj_name.upper()} - Frame {frame} (Proj Point)')
            axes[i, 0].axis('off')
            
            # 可视化 2: Segmentation Mask 中提取的目标颜色区域
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Extracted Mask (Color: {target_color})')
            axes[i, 1].axis('off')
            
            # 可视化 3: 用 Mask 抠出原始 RGB 中的物体
            extracted_rgb = np.zeros_like(rgb_img)
            extracted_rgb[mask] = rgb_img[mask]
            axes[i, 2].imshow(extracted_rgb)
            axes[i, 2].set_title(f'Clean Object RGB')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('extraction_result.png', dpi=300, bbox_inches='tight')
        self.dataset.close()

# =========== 运行入口 ===========
if __name__ == "__main__":
    # 替换为你的本地 hdf5 路径
    hdf5_path = "/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/episode0.hdf5"
    extractor = InteractionObjectExtractor(hdf5_path)
    extractor.visualize_extraction()