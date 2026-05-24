import h5py

def print_hdf5_structure(h5_path):
    print(f"========== 解析文件: {h5_path} ==========")
    with h5py.File(h5_path, 'r') as f:
        # 定义一个回调函数，只有遇到具体的数据集（Dataset）时才打印路径和形状
        def print_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"键名 (Key): {name:<30} | 形状 (Shape): {obj.shape} | 类型 (Type): {obj.dtype}")
        
        f.visititems(print_info)
    print("=========================================\n")

# 使用示例
print_hdf5_structure(r"/root/autodl-tmp/RoboTwin/data/place_can_basket/my_clean/data/episode0.hdf5")