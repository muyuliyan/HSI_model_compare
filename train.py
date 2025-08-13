# train.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.FCNN3D import FCNN3D, center_crop
from metrics.detector import evaluate_metrics
import argparse

# python train.py --dataset CAVE

# 加载patch数据
def load_patches(patch_dir, subdir='X'):
    all_patches = []
    shapes = set()
    for root, dirs, files in os.walk(patch_dir):
        if subdir and os.path.basename(root) != subdir:
            continue
        for fname in files:
            if fname.endswith('.npy'):
                fpath = os.path.join(root, fname)
                patches = np.load(fpath)  # [N, C, H, W]
                all_patches.append(patches)
                shapes.add(patches.shape[1])  # 记录C通道数
    if len(all_patches) == 0:
        raise RuntimeError(f"No .npy patch files found in {patch_dir}/{subdir}, please check your preprocessing output and path!")
    if len(shapes) > 1:
        print(f"Error: Found patches with different channel numbers: {shapes}")
        for root, dirs, files in os.walk(patch_dir):
            if subdir and os.path.basename(root) != subdir:
                continue
            for fname in files:
                if fname.endswith('.npy'):
                    fpath = os.path.join(root, fname)
                    arr = np.load(fpath)
                    print(f"{fpath}: shape {arr.shape}")
        raise RuntimeError("Patch channel numbers are inconsistent, please check your preprocessing!")
    all_patches = np.concatenate(all_patches, axis=0)  # [total_N, C, H, W]
    return all_patches

# 训练/测试集
def get_dataloaders(patch_dir, batch_size=8, test_ratio=0.1):
    patches = load_patches(patch_dir)  # [N, C, H, W]
    patches = torch.from_numpy(patches).float()
    # 假设输入为低分辨率，标签为高分辨率（此处示例直接用patch自身做标签，实际应用中需自行生成LR/HR对）
    inputs = patches.unsqueeze(1)  # [N, 1, C, H, W]
    targets = patches.unsqueeze(1) # [N, 1, C, H, W]
    dataset = TensorDataset(inputs, targets)
    test_len = int(len(dataset) * test_ratio)
    train_len = len(dataset) - test_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

# 训练评估
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in train_loader:
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
        for x, y in test_loader:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CAVE', help='选择数据集名称')
    args = parser.parse_args()
    patch_dir = f"./datasets/patches/{args.dataset}"

    # 参数
    batch_size = 8
    lr = 1e-3
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, test_loader = get_dataloaders(patch_dir, batch_size=batch_size)

    # 模型
    model = FCNN3D(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 训练与评估
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        metrics = test(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, " +
              ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

    # 可选：保存模型
    torch.save(model.state_dict(), ".\saved\fcnn3d_final.pth")