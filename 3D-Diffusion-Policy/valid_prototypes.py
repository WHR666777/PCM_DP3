import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class PrototypeVerifier:
    def __init__(self, prototype_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("⏳ 正在加载 DINOv2 模型...")
        
        # 使用你之前跑通的离线加载方式
        local_repo_path = '/root/autodl-tmp/dinov2'
        self.model = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"📥 正在加载图鉴库: {prototype_path}")
        # 加载图鉴 [K, 384] 并进行 L2 归一化 (计算余弦相似度必须)
        self.prototypes = torch.load(prototype_path, map_location=self.device)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1) 
        self.K = self.prototypes.shape[0]

        # 图像预处理 (注意：这里不需要 Resize，我们需要保留原图的长宽比)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def process_and_visualize(self, img_path, threshold=0.65):
        # 1. 读取并处理图像
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig, _ = img_rgb.shape
        
        # DINOv2 要求输入尺寸必须是 14 的倍数 (patch_size=14)
        H_new, W_new = (H_orig // 14) * 14, (W_orig // 14) * 14
        img_resized = cv2.resize(img_rgb, (W_new, H_new))
        
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device) # [1, 3, H, W]
        
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
        
        # 3. 计算余弦相似度 (内积)
        # dense_features: [1, 384, H_patch, W_patch]
        # prototypes: [K, 384]
        # 结果 sim_maps: [1, K, H_patch, W_patch]
        sim_maps = torch.einsum('b c h w, k c -> b k h w', dense_features, self.prototypes)
        
        # 4. 多原型融合 (Max-Pooling)
        max_sim, _ = torch.max(sim_maps, dim=1) # [1, H_patch, W_patch]
        max_sim = max_sim.squeeze().cpu().numpy()
        
        # 5. 上采样回原图分辨率
        max_sim_resized = cv2.resize(max_sim, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        
        # 6. 硬截断生成 Mask
        mask = max_sim_resized > threshold
        
        # 可视化
        self._plot_results(img_rgb, max_sim_resized, mask, threshold)

    def _plot_results(self, img, heatmap, mask, threshold):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原图
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 热力图
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Cosine Similarity Heatmap")
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Mask 结果图
        masked_img = img.copy()
        masked_img[~mask] = [0, 0, 0] # 将非目标区域涂黑
        axes[2].imshow(masked_img)
        axes[2].set_title(f"Hard Masked (Threshold > {threshold})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/root/autodl-tmp/RoboTwin/policy/my_DP3/validation_result.png', dpi=300)
        print("✅ 验证图像已保存至 validation_result.png")

if __name__ == "__main__":
    # 替换为你刚才生成的 pt 文件路径
    PROTO_PATH = "/root/autodl-tmp/RoboTwin/policy/my_DP3/DINOv2/prototype_can_k2.pt"
    # 找一张带目标物体的全尺寸环境图 (可以是含有杂物的图，或者专家数据某一帧的全图)
    TEST_IMG_PATH = "/root/autodl-tmp/RoboTwin/data/my_process_data/frame_000000.png" 
    
    verifier = PrototypeVerifier(PROTO_PATH)
    
    # 你可以修改阈值 threshold，看看设为多少能最完美地扣出物体
    verifier.process_and_visualize(TEST_IMG_PATH, threshold=0.35)