import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型定义 (必须与训练时完全一致)
# ==========================================
class DINOv2LinearProbe(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 替换为你的本地 DINOv2 路径
        local_repo_path = '/root/autodl-tmp/dinov2'
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"
        
        self.backbone = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        in_channels = 384 * 4
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            B, C, H, W = x.shape
            feat_h, feat_w = H // 14, W // 14
            layers = self.backbone.get_intermediate_layers(x, n=4, return_class_token=False)
            concat_features = torch.cat(layers, dim=-1)
            dense_features = concat_features.permute(0, 2, 1).reshape(B, 1536, feat_h, feat_w)
            
        logits = self.head(dense_features)
        return logits

# ==========================================
# 2. 推理引擎
# ==========================================
class SegmentationInferencer:
    def __init__(self, head_weight_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"⏳ 正在加载推理模型至 {self.device}...")
        
        self.model = DINOv2LinearProbe(num_classes=3).to(self.device)
        self.model.head.load_state_dict(torch.load(head_weight_path, map_location=self.device))
        self.model.eval()
        
        # 预处理：保持与训练时一致的均值和标准差
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self.target_size = (518, 518) # 必须是 14 的倍数

    @torch.no_grad()
    def predict(self, image_np):
        """
        image_np: 形状为 (H, W, 3) 的 RGB numpy 数组，值域 0-255
        返回: 形状为 (H, W) 的整数 Mask numpy 数组 (值为 0, 1, 2)
        """
        orig_h, orig_w = image_np.shape[:2]
        
        # 1. 预处理
        img_pil = Image.fromarray(image_np)
        img_resized = TF.resize(img_pil, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = self.normalize(img_tensor).unsqueeze(0).to(self.device) # [1, 3, 518, 518]
        
        # 2. 模型前向传播，得到低分辨率的 Logits (例如 37x37)
        logits = self.model(img_tensor) # [1, 3, 37, 37]
        
        # 3. 极其关键：将 Logits 用双线性插值放大回【原始图像尺寸】
        logits_upsampled = F.interpolate(
            logits, 
            size=(orig_h, orig_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 4. Argmax 提取最可能的类别
        mask_tensor = torch.argmax(logits_upsampled, dim=1).squeeze(0) # [orig_h, orig_w]
        
        return mask_tensor.cpu().numpy()

def postprocess_mask(mask_np, target_classes=[1, 2], dilate_kernel_size=5):
    """
    对 DINOv2 输出的多类别掩膜进行后处理：最大连通域过滤 + 形态学膨胀
    
    参数:
        mask_np: 原始预测的 numpy 数组，形状 (H, W)，值域 {0, 1, 2}
        target_classes: 需要进行处理的类别列表
        dilate_kernel_size: 膨胀核大小，数值越大，边缘向外扩张越多
    返回:
        processed_mask: 处理后的干净掩膜
    """
    # 建立一个全 0 (背景) 的空白画布
    processed_mask = np.zeros_like(mask_np)
    
    # 定义形态学膨胀的核 (使用椭圆形核，边缘更圆滑)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    
    for class_id in target_classes:
        # 1. 提取当前类别的二值化掩膜 (背景是0，当前类是1)
        binary_mask = (mask_np == class_id).astype(np.uint8)
        
        # 如果画面里根本没有预测出这个类别，直接跳过
        if not np.any(binary_mask):
            continue
            
        # 2. 寻找连通域 (connectivity=8 考虑对角线相邻)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # 3. 最大连通域过滤
        # num_labels 至少是 1（代表全黑背景）。大于 1 才有目标色块。
        if num_labels > 1:
            # 获取所有色块的面积 (去除第0个，因为那是背景的面积)
            areas = stats[1:, cv2.CC_STAT_AREA]
            # 找到面积最大的色块索引 (+1 是因为我们剔除了背景)
            largest_label = np.argmax(areas) + 1
            
            # 生成只有最大连通域的纯净二值掩膜
            clean_binary = (labels == largest_label).astype(np.uint8)
        else:
            clean_binary = binary_mask
            
        # 4. 形态学膨胀 (把瘦了一圈的边缘扩张回来)
        # iterations=1 表示操作一次
        dilated_binary = cv2.dilate(clean_binary, kernel, iterations=1)
        
        # 5. 将处理好的类别写回总掩膜中
        processed_mask[dilated_binary == 1] = class_id
        
    return processed_mask
    
import h5py
import cv2
from tqdm import tqdm

# ==========================================
# 3. 视频生成逻辑 (在你的类定义之后添加)
# ==========================================

def create_mask_video_from_hdf5(hdf5_path, head_weight_path, output_video_path, fps=20):
    """
    读取 HDF5 中的压缩 RGB 流，生成分割掩膜视频
    """
    # 1. 初始化推理引擎
    inferencer = SegmentationInferencer(head_weight_path)
    
    # 2. 打开 HDF5 数据集
    # 根据你之前的代码逻辑，路径为 'observation/head_camera/rgb'
    db = h5py.File(hdf5_path, 'r')
    rgbs = db['observation/head_camera/rgb']
    num_frames = rgbs.shape[0]
    
    # 3. 预读取第一帧以确定视频尺寸
    first_raw = rgbs[0]
    first_img = cv2.imdecode(np.frombuffer(first_raw, np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = first_img.shape
    
    # 4. 初始化视频写入器
    # 使用 mp4v 编码，保存为 .mp4 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # 定义类别颜色映射 (BGR 格式)
    # 0=背景(黑), 1=可乐(红), 2=篮子(蓝)
    color_palette = np.array([
        [0, 0, 0],       # Background
        [0, 0, 255],     # Can (Red in BGR)
        [255, 128, 0]    # Basket (Blue-ish in BGR)
    ], dtype=np.uint8)

    print(f"🎬 开始处理 HDF5 视频流，共 {num_frames} 帧...")

    try:
        for i in tqdm(range(num_frames)):
            # A. 解码图像
            raw_bytes = rgbs[i]
            img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # B. 执行 DINOv2 推理得到 Mask (0, 1, 2)
            mask = inferencer.predict(img_rgb)
            clean_mask = postprocess_mask(mask, target_classes=[1, 2], dilate_kernel_size=3)
            # C. 渲染可视化效果
            # 方法 1：纯掩膜彩色视频
            # colored_mask = color_palette[mask]
            
            # 方法 2：半透明叠加效果 (推荐，方便观察分割精度)
            mask_rgb = color_palette[clean_mask]
            overlay = cv2.addWeighted(img_bgr, 0.6, mask_rgb, 0.4, 0)
            
            # D. 写入帧
            video_writer.write(overlay)

    finally:
        # 5. 释放资源
        video_writer.release()
        db.close()
        print(f"✅ 掩膜视频已生成: {output_video_path}")

# ==========================================
# 4. 执行区域
# ==========================================
if __name__ == "__main__":
    # 配置你的路径
    HDF5_FILE = "/root/autodl-tmp/RoboTwin/data/place_can_basket/demo_randomized/data/episode24.hdf5"
    WEIGHT_FILE = "/root/autodl-tmp/RoboTwin/policy/my_DP3/3D-Diffusion-Policy/checkpoints/dinov2_linear_head.pth"
    OUTPUT_FILE = "segmentation_output.mp4"
    
    create_mask_video_from_hdf5(HDF5_FILE, WEIGHT_FILE, OUTPUT_FILE)