# trainINP.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from metrics.detector import evaluate_metrics
import argparse
from tqdm import tqdm

# Dataset: 支持 mask 缺失
class InpaintPatchDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dir, mask_ratio=0.2):
        self.file_list = []
        for root, dirs, files in os.walk(patch_dir):
            for fname in files:
                if fname.endswith('.npy'):
                    fpath = os.path.join(root, fname)
                    arr = np.load(fpath, mmap_mode='r')
                    n = arr.shape[0]
                    for i in range(n):
                        self.file_list.append((fpath, i))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npy patch files found in {patch_dir}")
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath, i = self.file_list[idx]
        patch = np.load(fpath, mmap_mode='r')[i].copy()  # [C, H, W]
        # 随机 mask
        mask = np.random.rand(*patch.shape) > self.mask_ratio
        masked_patch = patch * mask
        return torch.from_numpy(masked_patch).float(), torch.from_numpy(patch).float(), torch.from_numpy(mask.astype(np.float32))

def get_dataloaders(patch_dir, batch_size=8, test_ratio=0.1, mask_ratio=0.2):
    dataset = InpaintPatchDataset(patch_dir, mask_ratio=mask_ratio)
    test_len = int(len(dataset) * test_ratio)
    train_len = len(dataset) - test_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

def import_model(model_name, channels):
    if model_name == 'CNN3D':
        from INPmodels.CNN3D import HSI_3D_CNN
        return HSI_3D_CNN()  # 你的实现不需要 bands 参数
    elif model_name == 'HSISRN':
        from INPmodels.HSISRN import HSI_SRN
        return HSI_SRN(in_channels=channels)
    elif model_name == 'SDeCNN':
        from INPmodels.SDeCNN import HSI_SDeCNN
        return HSI_SDeCNN(bands=channels)
    elif model_name == 'Transformer':
        from INPmodels.Transformer import HSI_Transformer
        return HSI_Transformer(bands=channels)
    else:
        raise ValueError('未知模型')

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x_masked, x_gt, mask in tqdm(train_loader, desc='Train', leave=False):
        x_masked, x_gt, mask = x_masked.to(device), x_gt.to(device), mask.to(device)
        pred = model(x_masked)
        # 只对已知像素计算损失
        loss = loss_fn(pred * mask, x_gt * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for x_masked, x_gt, mask in tqdm(test_loader, desc='Test', leave=False):
            x_masked, x_gt, mask = x_masked.to(device), x_gt.to(device), mask.to(device)
            pred = model(x_masked)
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = x_gt.squeeze().cpu().numpy()
            metrics = evaluate_metrics(pred_np, gt_np, ratio=1)
            metrics_list.append(metrics)
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, required=True, help='patch数据目录')
    parser.add_argument('--model', type=str, required=True, help='模型名：3DCNN, HSISRN, SDeCNN, Transformer')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='随机mask比例')
    args = parser.parse_args()

    batch_size = 2
    lr = 1e-3
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(args.patch_dir, batch_size=batch_size, mask_ratio=args.mask_ratio)
    channels = next(iter(train_loader))[0].shape[1]
    model = import_model(args.model, channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        metrics = test(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    os.makedirs('./saved', exist_ok=True)
    torch.save(model.state_dict(), f'./saved/{args.model.lower()}_inp_final.pth')