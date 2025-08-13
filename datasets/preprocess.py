# preprocess.py
import os
import numpy as np
import torch

def extract_patches(img, patch_size, stride):
    """
    从高光谱图像中提取小块patch
    img: numpy array, shape [C, H, W]
    patch_size: int or tuple, patch的空间尺寸
    stride: int, 滑动步长
    返回: [N, C, patch_H, patch_W]
    """
    C, H, W = img.shape
    ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
    patches = []
    for i in range(0, H - ph + 1, stride):
        for j in range(0, W - pw + 1, stride):
            patch = img[:, i:i+ph, j:j+pw]
            patches.append(patch)
    if len(patches) == 0:
        raise ValueError(f"No patches extracted! Check patch_size={patch_size}, stride={stride}, image shape={img.shape}")
    return np.stack(patches, axis=0)

def normalize(img):
    """
    简单归一化到[0,1]
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

def preprocess_hsi_dataset(data_dir, out_dir, patch_size=32, stride=16):
    """
    针对高光谱数据集的通用预处理，支持多级子目录
    data_dir: 原始数据文件夹
    out_dir: 预处理后保存文件夹
    patch_size: patch大小
    stride: 滑动步长
    """
    for root, dirs, files in os.walk(data_dir):
        rel_dir = os.path.relpath(root, data_dir)
        save_dir = os.path.join(out_dir, rel_dir) if rel_dir != '.' else out_dir
        os.makedirs(save_dir, exist_ok=True)
        for fname in files:
            if not fname.endswith(('.mat', '.npy', '.tif')):
                continue
            fpath = os.path.join(root, fname)
            # 加载数据

            if fname.endswith('.npy'):
                img = np.load(fpath)
            elif fname.endswith('.mat'):
                try:
                    import scipy.io
                    mat = scipy.io.loadmat(fpath)
                    # 自动检测主数据 key
                    exclude_keys = {'__header__', '__version__', '__globals__'}
                    data_keys = [k for k in mat.keys() if k not in exclude_keys]
                    if not data_keys:
                        print(f"Warning: {fname} has no valid data key, skip.")
                        continue
                    if len(data_keys) > 1:
                        print(f"Info: {fname} has multiple keys {data_keys}, use the first one: {data_keys[0]}")
                    img = mat[data_keys[0]]
                    img = np.array(img)
                    # 自动适配 shape 到 [C, H, W]
                    if img.ndim == 3:
                        # [H, W, C] -> [C, H, W]，只要最后一个维度小于100就转置
                        if img.shape[2] < 100:
                            img = img.transpose(2, 0, 1)
                        # [C, H, W]，只要第一个维度小于100就直接用
                        elif img.shape[0] < 100:
                            pass
                        else:
                            print(f"Warning: {fname} shape {img.shape} is not recognized as HWC or CHW, skip.")
                            continue
                    else:
                        print(f"Warning: {fname} data shape {img.shape} is not 3D, skip.")
                        continue
                except Exception as e:
                    # 只处理特定异常
                    if isinstance(e, (NotImplementedError, ValueError)):
                        import h5py
                        with h5py.File(fpath, 'r') as f:
                            # 自动找第一个主数据 key
                            data_keys = [k for k in f.keys()]
                            if not data_keys:
                                print(f"Warning: {fname} has no valid data key (h5), skip.")
                                continue
                            if len(data_keys) > 1:
                                print(f"Info: {fname} (h5) has multiple keys {data_keys}, use the first one: {data_keys[0]}")
                            img = f[data_keys[0]][()]
                            img = np.array(img)
                            # HDF5 读出来一般是 (H, W, C) 或 (C, H, W)，和前面一样判断 shape
                            if img.ndim == 3:
                                if img.shape[2] < 100:
                                    img = img.transpose(2, 0, 1)
                                elif img.shape[0] < 100:
                                    pass
                                else:
                                    print(f"Warning: {fname} (h5) shape {img.shape} is not recognized as HWC or CHW, skip.")
                                    continue
                            else:
                                print(f"Warning: {fname} (h5) data shape {img.shape} is not 3D, skip.")
                                continue
                    else:
                        print(f"Error loading {fname}: {e}")
                        continue
            elif fname.endswith('.tif'):
                import tifffile
                img = tifffile.imread(fpath)
                if img.shape[0] < 10:
                    img = img.transpose(2, 0, 1)
            else:
                continue

            img = normalize(img)
            C, H, W = img.shape
            if H < patch_size or W < patch_size:
                # 不切分，直接保存整个图像
                out_path = os.path.join(save_dir, fname.replace('.', '_') + '_full.npy')
                np.save(out_path, img[None, ...])  # 保持和patches一致的4D格式 [1, C, H, W]
                print(f"{os.path.join(rel_dir, fname)} -> full image saved to {out_path}")
                continue
            patches = extract_patches(img, patch_size, stride)
            # 保存patches，保留子目录结构
            out_path = os.path.join(save_dir, fname.replace('.', '_') + f'_patch{patch_size}.npy')
            np.save(out_path, patches)
            print(f"{os.path.join(rel_dir, fname)} -> {patches.shape} patches saved to {out_path}")

def auto_preprocess_all():
    configs = [
        {"name": "CAVE", "data_dir": "./data/CAVE", "out_dir": "datasets/patches/CAVE", "patch_size": 32, "stride": 16},
        {"name": "PaviaU", "data_dir": "./data/PaviaU", "out_dir": "datasets/patches/PaviaU", "patch_size": 32, "stride": 16},
        {"name": "IndianPines", "data_dir": "./data/IndianPines", "out_dir": "datasets/patches/IndianPines", "patch_size": 16, "stride": 8},
        {"name": "Harvard", "data_dir": "./data/Harvard", "out_dir": "datasets/patches/Harvard", "patch_size": 32, "stride": 16},
        {"name": "Chikusei", "data_dir": "./data/Chikusei", "out_dir": "datasets/patches/Chikusei", "patch_size": 32, "stride": 16},
    ]
    for cfg in configs:
        print(f"Processing {cfg['name']} ...")
        preprocess_hsi_dataset(
            data_dir=cfg["data_dir"],
            out_dir=cfg["out_dir"],
            patch_size=cfg["patch_size"],
            stride=cfg["stride"]
        )

if __name__ == "__main__":
    auto_preprocess_all()