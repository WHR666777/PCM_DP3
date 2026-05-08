# semantic_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. DINOv2 + 线性分类头模型定义 (与训练时保持一致)
# ==========================================
class DINOv2LinearProbe(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 注意：你需要根据实际存放位置修改以下两个路径
        local_repo_path = '/root/autodl-tmp/dinov2'               # DINOv2 仓库本地路径
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"  # 预训练权重
        
        self.backbone = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # 拼接 4 个中间层，每层 384 维，共 384*4 = 1536
        in_channels = 384 * 4
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            B, C, H, W = x.shape
            feat_h, feat_w = H // 14, W // 14
            layers = self.backbone.get_intermediate_layers(x, n=4, return_class_token=False)
            concat_features = torch.cat(layers, dim=-1)                     # (B, N, 1536)
            dense_features = concat_features.permute(0, 2, 1).reshape(B, 1536, feat_h, feat_w)
        logits = self.head(dense_features)
        return logits


# ==========================================
# 2. 语义点云提取器 (符合 deploy_policy.py 的调用接口)
# ==========================================
class SemanticPointExtractor:
    def __init__(self, head_weight_path: str, device: str = "cuda", postprocess: bool = True):
        """
        head_weight_path: 训练好的线性头权重文件路径
        device: 推理设备，默认 'cuda'
        postprocess: 是否对掩膜进行最大连通域+膨胀后处理
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.postprocess = postprocess
        print(f"⏳ 正在加载 DINOv2 语义分割模型至 {self.device}...")
        
        self.model = DINOv2LinearProbe(num_classes=3).to(self.device)
        self.model.head.load_state_dict(torch.load(head_weight_path, map_location=self.device))
        self.model.eval()
        
        # 预处理参数（与训练时相同）
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std  = [0.229, 0.224, 0.225]
        self.target_size = (518, 518)  # 必须是 14 的倍数
        
        if self.postprocess:
            # 形态学膨胀核，可根据需要调整大小
            self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @torch.no_grad()
    def predict(self, image_np: np.ndarray) -> np.ndarray:
        """
        输入: RGB 图像，形状 (H, W, 3)，dtype uint8，值域 [0, 255]
        输出: 整数掩膜，形状 (H, W)，dtype int64，值为 0/1/2 对应背景/目标1/目标2
        """
        orig_h, orig_w = image_np.shape[:2]
        
        # 1. 预处理 -> Tensor
        img_pil = Image.fromarray(image_np)
        img_resized = TF.resize(img_pil, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = TF.normalize(img_tensor, mean=self.normalize_mean, std=self.normalize_std)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, 3, 518, 518)
        
        # 2. 推理低分辨率 logits
        logits = self.model(img_tensor)  # (1, num_classes, H/14, W/14)
        
        # 3. 上采样至原始图像尺寸
        logits_upsampled = F.interpolate(
            logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        
        # 4. 取 argmax 获得类别掩膜
        mask_tensor = torch.argmax(logits_upsampled, dim=1).squeeze(0)  # (orig_h, orig_w)
        mask_np = mask_tensor.cpu().numpy()
        
        # 5. 可选后处理（最大连通域 + 膨胀）
        if self.postprocess:
            mask_np = self._postprocess_mask(mask_np, target_classes=[1, 2])
        
        return mask_np

    def _postprocess_mask(self, mask_np: np.ndarray, target_classes=[1, 2]) -> np.ndarray:
        """
        对每个目标类别执行：
          - 保留最大连通域（去除噪点）
          - 形态学膨胀（补偿上采样带来的边缘收缩）
        """
        processed_mask = np.zeros_like(mask_np, dtype=np.int64)
        kernel = self.dilate_kernel
        
        for class_id in target_classes:
            binary = (mask_np == class_id).astype(np.uint8)
            if not np.any(binary):
                continue
                
            # 连通域分析，保留最大区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = np.argmax(areas) + 1
                clean_binary = (labels == largest_label).astype(np.uint8)
            else:
                clean_binary = binary
                
            # 膨胀
            dilated = cv2.dilate(clean_binary, kernel, iterations=1)
            processed_mask[dilated == 1] = class_id
            
        return processed_mask