import os
os.environ["OPENBLAS_NUM_THREADS"] = "15"
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["MKL_NUM_THREADS"] = "15"
os.environ["NUMEXPR_NUM_THREADS"] = "15"
import glob
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class PrototypeBuilder:
    def __init__(self):
        print("⏳ 正在从本地数据盘完全离线加载 DINOv2 模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        local_repo_path = '/root/autodl-tmp/dinov2'
        
        if not os.path.exists(local_repo_path):
            raise RuntimeError(f"❌ 找不到源码目录: {local_repo_path}")
            
        self.model = torch.hub.load(local_repo_path, 'dinov2_vits14_reg', source='local', pretrained=False)
        
        weight_path = "/root/autodl-tmp/dinov2_vits14_reg4_pretrain.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"❌ 找不到权重文件: {weight_path}")
            
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, image_dir):
        """遍历文件夹内的图片，提取 DINOv2 多层密集融合特征"""
        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not img_paths:
            raise ValueError(f"❌ 目录 {image_dir} 中没有找到图片！")
            
        print(f"🔍 开始提取 {os.path.basename(image_dir)} 的特征 (共 {len(img_paths)} 张)...")
        features_list = []
        
        for path in tqdm(img_paths, desc="提取特征"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            
            tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # 提取最后 4 层特征 (不包含 cls token)
            # 返回的是一个 tuple，包含 4 个 tensor，每个形状为 [1, 256, 384] (基于 224x224 输入)
            layers = self.model.get_intermediate_layers(tensor, n=4, return_class_token=False)
            
            # 在特征维度 (dim=-1) 拼接这 4 层的特征，得到 [1, 256, 1536] 的张量
            concat_features = torch.cat(layers, dim=-1)
            
            # 将 256 个 patch 还原为 16x16 的网格空间排列
            patch_grid = concat_features.reshape(16, 16, -1)
            
            # 裁剪中心 6x6 区域 (确保全是物体本体，排除边缘背景)
            # 如果你的裁剪图里物体更小，可以改得更紧凑比如 [6:10, 6:10, :]
            center_patches = patch_grid[5:11, 5:11, :].reshape(-1, 1536)
            
            features_list.append(center_patches.cpu().numpy())
            
        # 注意：这里需要用 vstack 把所有图片的 patch 堆叠成一个巨大的二维矩阵用于 K-Means
        return np.vstack(features_list)

    def build_and_save(self, input_dir, output_path, k_values=[5], cache_dir=None):
        """支持特征缓存，并能一次性生成多个 K 值的图鉴"""
        object_name = os.path.basename(input_dir)
        
        # 1. 缓存拦截机制：检查是否已经有提取好的特征矩阵
        feature_cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            feature_cache_path = os.path.join(cache_dir, f"{object_name}_features.npy")
            
        if feature_cache_path and os.path.exists(feature_cache_path):
            print(f"⚡ 发现本地缓存！直接秒读特征: {feature_cache_path}")
            features = np.load(feature_cache_path)
        else:
            # 没有缓存，老老实实提取
            features = self.extract_features(input_dir)
            # 提取完立刻存盘，下次直接白嫖
            if feature_cache_path:
                np.save(feature_cache_path, features)
                print(f"💾 特征已缓存至: {feature_cache_path}")
        
        # 2. 极速循环聚类：几秒钟内跑完所有你想要的 K 值
        for k in k_values:
            print(f"🧠 正在进行 K-Means 聚类 (K={k})...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(features)
            
            prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            
            # 动态生成带 K 值的保存路径，如 prototype_can_k5.pt, prototype_can_k10.pt
            base_name, ext = os.path.splitext(output_path)
            final_output_path = f"{base_name}_k{k}{ext}"
            
            torch.save(prototypes, final_output_path)
            print(f"✅ 图鉴库 (K={k}) 已固化并保存至: {final_output_path}")

if __name__ == "__main__":
    IMG_ROOT = "/root/autodl-tmp/RoboTwin/data/dino_prototypes_data/"
    CAN_DIR = os.path.join(IMG_ROOT, "can")
    BASKET_DIR = os.path.join(IMG_ROOT, "basket")
    
    OUTPUT_DIR = "/root/autodl-tmp/RoboTwin/policy/my_DP3/DINOv2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 建立一个专门存特征数组的缓存文件夹
    CACHE_DIR = os.path.join(OUTPUT_DIR, "feature_cache")
    
    # 实例化 (不再需要传入 K 值)
    builder = PrototypeBuilder()
    
    # ================= 极速调参区 =================
    # 你想测试哪些 K 值，直接写在列表里！
    can_k_test = [1, 2, 3, 4]  
    basket_k_test = [3, 5, 8]
    
    print("\n" + "="*40)
    print("开始构建【可乐】图鉴...")
    builder.build_and_save(CAN_DIR, os.path.join(OUTPUT_DIR, "prototype_can.pt"), k_values=can_k_test, cache_dir=CACHE_DIR)
    
    print("\n" + "="*40)
    print("开始构建【篮子】图鉴...")
    builder.build_and_save(BASKET_DIR, os.path.join(OUTPUT_DIR, "prototype_basket.pt"), k_values=basket_k_test, cache_dir=CACHE_DIR)
    
    print("\n🎉 全部 K 值图鉴生成完毕！尽情去验证脚本里测试吧！")