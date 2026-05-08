import os
import torch
import cv2
import h5py
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

class DynamicVideoVerifier:
    def __init__(self, prototype_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("⏳ 正在离线加载 DINOv2 模型...")
        
        # 本地离线加载 DINOv2
        local_repo_path = '/root/autodl-tmp/dinov2'
        self.model = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"📥 正在加载图鉴库: {prototype_path}")
        self.prototypes = torch.load(prototype_path, map_location=self.device)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1) 
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def process_episode_to_video(self, hdf5_path, output_video_path, threshold=0.65, max_frames=None):
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"❌ 找不到 HDF5 文件: {hdf5_path}")

        # 1. 打开 HDF5 文件
        dataset = h5py.File(hdf5_path, 'r')
        rgbs = dataset['observation/head_camera/rgb']
        num_frames = rgbs.shape[0]
        
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
            
        print(f"🎥 开始处理视频，共 {num_frames} 帧 (阈值 Threshold={threshold})...")
        
        video_writer = None
        
        for i in tqdm(range(num_frames)):
            # 2. 读取并解码 RGB 图像 (BGR 格式供 OpenCV 使用)
            raw_bytes = rgbs[i]
            img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H_orig, W_orig, _ = img_rgb.shape
            
            # 3. DINOv2 处理 (尺寸补齐为 14 的倍数)
            H_new, W_new = (H_orig // 14) * 14, (W_orig // 14) * 14
            img_resized = cv2.resize(img_rgb, (W_new, H_new))
            
            tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
            
            # 2. 【核心步骤】：提取多层密集融合特征 (与建图鉴时保持一致)
            # 获取最后 4 层特征，不包含 cls token
            layers = self.model.get_intermediate_layers(tensor, n=4, return_class_token=False)
            
            # 将 4 层的特征在通道维度拼接，得到 [1, H_patch * W_patch, 1536]
            concat_features = torch.cat(layers, dim=-1)
            
            # 变形为二维特征图 [1, 1536, H_patch, W_patch] (注意维度变成了 1536)
            H_patch, W_patch = H_new // 14, W_new // 14
            dense_features = concat_features.permute(0, 2, 1).reshape(1, 1536, H_patch, W_patch)
            
            # 对密集特征进行 L2 归一化 (保留此行不变)
            dense_features = F.normalize(dense_features, p=2, dim=1)
            
            sim_maps = torch.einsum('b c h w, k c -> b k h w', dense_features, self.prototypes)
            max_sim, _ = torch.max(sim_maps, dim=1)
            max_sim = max_sim.squeeze().cpu().numpy()
            
            # 5. 上采样回原图分辨率
            max_sim_resized = cv2.resize(max_sim, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            
            # 6. OpenCV 极速生成可视化图像
            # --- 原图 (加入帧号文字) ---
            img_viz = img_bgr.copy()
            cv2.putText(img_viz, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # --- 热力图 ---
            # 将余弦相似度 [-1, 1] 截断到 [0, 1] 并映射为颜色
            sim_norm = np.clip(max_sim_resized, 0, 1)
            heatmap_color = cv2.applyColorMap((sim_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # --- 截断掩码图 ---
            mask = max_sim_resized > threshold
            masked_img = img_bgr.copy()
            masked_img[~mask] = [0, 0, 0] # 涂黑非目标区域
            
            # 7. 画面拼接 (左：原图，中：热力图，右：掩码图)
            combined_frame = np.hstack([img_viz, heatmap_color, masked_img])
            
            # 8. 初始化 VideoWriter (仅在第一帧)
            if video_writer is None:
                h_comb, w_comb, _ = combined_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # 设定 20 FPS，你可以根据需要调整
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w_comb, h_comb))
                
            video_writer.write(combined_frame)

        # 释放资源
        video_writer.release()
        dataset.close()
        print(f"✅ 动态验证视频已生成: {output_video_path}")


if __name__ == "__main__":
    # 配置图鉴路径和你要验证的 HDF5 专家数据
    PROTO_PATH = "/root/autodl-tmp/RoboTwin/policy/my_DP3/DINOv2/prototype_basket_k3.pt"
    HDF5_PATH = "/root/autodl-tmp/RoboTwin/data/place_can_basket/demo_randomized/data/episode13.hdf5"
    # 输出视频名称
    OUTPUT_VIDEO = "/root/autodl-tmp/RoboTwin/policy/my_DP3/validation_head_camera.mp4"
    
    verifier = DynamicVideoVerifier(PROTO_PATH)
    
    # 执行视频生成 (你可以调整阈值，比如 0.60 或 0.65)
    verifier.process_episode_to_video(HDF5_PATH, OUTPUT_VIDEO, threshold=0.45)