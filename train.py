# train.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from metrics.detector import evaluate_metrics
import argparse
from tqdm import tqdm

# python train.py --patch_dir ./my/custom/path --subdir X

# 懒加载Dataset
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dir):
        self.file_list = []
        self.shapes = set()
        for root, dirs, files in os.walk(patch_dir):
            for fname in files:
                if fname.endswith('.npy'):
                    fpath = os.path.join(root, fname)
                    arr = np.load(fpath, mmap_mode='r')
                    n = arr.shape[0]
                    c = arr.shape[1]
                    self.shapes.add(c)
                    for i in range(n):
                        self.file_list.append((fpath, i))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npy patch files found in {patch_dir}, please check your preprocessing output and path!")
        if len(self.shapes) > 1:
            print(f"Error: Found patches with different channel numbers: {self.shapes}")
            for root, dirs, files in os.walk(patch_dir):
                for fname in files:
                    if fname.endswith('.npy'):
                        fpath = os.path.join(root, fname)
                        arr = np.load(fpath)
                        print(f"{fpath}: shape {arr.shape}")
            raise RuntimeError("Patch channel numbers are inconsistent, please check your preprocessing!")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath, i = self.file_list[idx]
        arr = np.load(fpath, mmap_mode='r')
        patch = arr[i].copy()  # 保证是可写的
        return patch

# 训练/测试集

def get_dataloaders(patch_dir, batch_size=8, test_ratio=0.1):
    dataset = PatchDataset(patch_dir)
    # 自动生成超分任务的LR/HR对：输入为下采样再上采样的低分辨率patch，标签为原始高分辨率patch
    import torch.nn.functional as F
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, ratio=4):
            self.base_dataset = base_dataset
            self.ratio = ratio
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            patch = self.base_dataset[idx]  # numpy [C, H, W]
            patch = torch.from_numpy(patch).float()  # [C, H, W]
            hr = patch.unsqueeze(0)  # [1, C, H, W]
            # 生成LR: 先下采样再上采样
            lr = F.interpolate(hr, scale_factor=1/self.ratio, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
            return lr_up, hr  # [1, C, H, W], [1, C, H, W]
    pair_dataset = PairDataset(dataset, ratio=4)
    test_len = int(len(pair_dataset) * test_ratio)
    train_len = len(pair_dataset) - test_len
    train_set, test_set = random_split(pair_dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

# 训练评估
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        # 裁剪对齐
        pred_crop = center_crop(pred, (y.shape[-2], y.shape[-1]))
        y_crop = center_crop(y, (pred_crop.shape[-2], pred_crop.shape[-1]))
        loss = loss_fn(pred_crop, y_crop)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Test', leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_crop = center_crop(pred, (y.shape[-2], y.shape[-1]))
            y_crop = center_crop(y, (pred_crop.shape[-2], pred_crop.shape[-1]))
            # 转为numpy [C, H, W]
            pred_np = pred_crop.squeeze().cpu().numpy()
            y_np = y_crop.squeeze().cpu().numpy()
            metrics = evaluate_metrics(pred_np, y_np, ratio=4)
            metrics_list.append(metrics)
    # 计算平均指标
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics

def import_model_and_crop(model_name):
    if model_name == 'FCNN3D':
        from models.FCNN3D import FCNN3D, center_crop
        return FCNN3D, center_crop
    elif model_name == 'MSSR':
        from models.MSSR import MSSR, center_crop
        return MSSR, center_crop
    elif model_name == 'RDN':
        from models.RDN import RDN_HSI, center_crop
        return RDN_HSI, center_crop
    elif model_name == 'Transformer':
        from models.Transformer import HSI_SR_Transformer, center_crop
        return HSI_SR_Transformer, center_crop
    elif model_name == 'PanNet':
        from models.PanNet import PanNet, center_crop
        return PanNet, center_crop
    else:
        raise ValueError('未知模型')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, required=True, help='指定patch数据的根目录或具体子目录，如datasets/patches/Chikusei 或 datasets/patches/Chikusei/X')
    parser.add_argument('--model', type=str, required=True, help='选择模型：FCNN3D, MSSR, RDN, Transformer, PanNet')
    args = parser.parse_args()
    patch_dir = args.patch_dir

    # 参数
    batch_size = 2
    lr = 1e-3
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, test_loader = get_dataloaders(patch_dir, batch_size=batch_size)

    # 模型
    channels = next(iter(train_loader))[1].shape[1]
    ModelClass, center_crop = import_model_and_crop(args.model)
    if args.model == 'FCNN3D':
        model = ModelClass(in_channels=1).to(device)
    elif args.model == 'MSSR':
        model = ModelClass(in_channels=1, out_channels=channels, scale_factor=4).to(device)
    elif args.model == 'RDN':
        model = ModelClass(in_channels=1, num_blocks=4, growth_rate=8, scale_factor=4).to(device)
    elif args.model == 'Transformer':
        model = ModelClass(in_channels=1, scale_factor=4).to(device)
    elif args.model == 'PanNet':
        model = ModelClass(in_ms_channels=channels).to(device)
    else:
        raise ValueError("未知模型")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 训练与评估
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        metrics = test(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, " +
              ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

    os.makedirs('./saved', exist_ok=True)
    torch.save(model.state_dict(), "./saved/fcnn3d_final.pth")