import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 数据集定义 (Dataset)
# ==========================================
class RoboTwinSegDataset(Dataset):
    def __init__(self, root_dir, target_size=(518, 518)):
        """
        target_size: 必须是 14 的倍数。518/14 = 37。
        """
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 获取所有图片路径
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        self.target_size = target_size
        
        # DINOv2 官方要求的标准化参数
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, filename)

        # 1. 读取 RGB 图像
        image = Image.open(img_path).convert('RGB')
        # 2. 读取 Mask (必须保持原始单通道，千万不能转 RGB)
        mask = Image.open(mask_path)

        # 3. 尺寸对齐处理
        # 图像使用双线性插值 (Bilinear)
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        # Mask 必须使用最近邻插值 (Nearest) 确保 0,1,2 不被破坏
        # 特别注意：我们将 Mask 直接缩小 14 倍，使其与 DINOv2 输出的特征网格对齐
        feat_h, feat_w = self.target_size[0] // 14, self.target_size[1] // 14
        mask = TF.resize(mask, (feat_h, feat_w), interpolation=TF.InterpolationMode.NEAREST)

        # 4. 转为 Tensor
        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        
        # Mask 转为 LongTensor，不要除以 255！
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask_tensor

# ==========================================
# 2. 模型定义 (Frozen DINOv2 + 1x1 Conv)
# ==========================================
class DINOv2LinearProbe(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print("⏳ 正在加载带有 Registers 的 DINOv2 模型...")
        
        # 请根据你的实际路径修改这里
        local_repo_path = '/root/autodl-tmp/dinov2'
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"
        
        self.backbone = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
        # 🚨 核心：彻底冻结骨干网络，不计算梯度
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # 提取最后 4 层特征，每层维度是 384，拼接后为 1536
        in_channels = 384 * 4
        
        # 轻量级分割头：1x1 卷积，参数量极小
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 骨干网络不计算梯度，节省显存并加快速度
        with torch.no_grad():
            B, C, H, W = x.shape
            feat_h, feat_w = H // 14, W // 14
            
            # 获取最后 4 层的特征
            layers = self.backbone.get_intermediate_layers(x, n=4, return_class_token=False)
            
            # 拼接 4 层特征 -> [B, N_patches, 1536]
            concat_features = torch.cat(layers, dim=-1)
            
            # Reshape 成二维网格 [B, 1536, feat_h, feat_w]
            dense_features = concat_features.permute(0, 2, 1).reshape(B, 1536, feat_h, feat_w)
            
        # 送入可训练的线性头
        logits = self.head(dense_features)
        return logits

# ==========================================
# 3. 训练主循环
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用计算设备: {device}")
    
    # 路径配置
    DATASET_ROOT = "/root/autodl-tmp/RoboTwin/data/dino_linear_probe_data" # 请替换为你的数据集实际路径
    BATCH_SIZE = 16
    EPOCHS = 30  # 因为只有 1x1 卷积，收敛非常快，通常 20-50 个 Epoch 足够
    LR = 1e-3

    # 初始化数据集和 DataLoader
    dataset = RoboTwinSegDataset(root_dir=DATASET_ROOT, target_size=(518, 518))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"📦 数据集加载成功，共 {len(dataset)} 张训练全景图。")

    # 初始化模型
    model = DINOv2LinearProbe(num_classes=3).to(device)
    
    # 【可选加速】如果你的 PyTorch >= 2.0，解除下面这行的注释可以提速
    # model = torch.compile(model)
    
    # 设定类别权重 (Class Weights) 应对极度不平衡问题
    # 背景(0)的像素极多，给较小的权重；可乐(1)和篮子(2)给较大的权重
    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 注意：只将 head 的参数传给优化器！
    optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)

    # 开始训练
    for epoch in range(EPOCHS):
        model.head.train() # 只需让 head 处于 train 模式
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device) # shape: [B, 37, 37]

            optimizer.zero_grad()
            
            # logits shape: [B, 3, 37, 37]
            logits = model(images)
            
            # CrossEntropyLoss 自动处理 logits 到 mask 的映射
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")

    # 保存训练好的线性头权重
    os.makedirs('checkpoints', exist_ok=True)
    save_path = "checkpoints/dinov2_linear_head.pth"
    torch.save(model.head.state_dict(), save_path)
    print(f"🎉 训练完成！权重已保存至 {save_path}")
    print("推理时，只需加载这个 head，对模型输出使用双线性插值放大到原图尺寸即可！")

if __name__ == "__main__":
    train()