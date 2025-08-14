import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from models.GAN import HSIGenerator, HSIPatchDiscriminator3D
from metrics.detector import evaluate_metrics
import argparse
from tqdm import tqdm

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

def get_dataloaders(patch_dir, batch_size=8, test_ratio=0.1):
    dataset = PatchDataset(patch_dir)
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            patch = self.base_dataset[idx]
            patch = torch.from_numpy(patch).float()
            return patch.unsqueeze(0), patch.unsqueeze(0)  # [1, C, H, W], [1, C, H, W]
    pair_dataset = PairDataset(dataset)
    test_len = int(len(pair_dataset) * test_ratio)
    train_len = len(pair_dataset) - test_len
    train_set, test_set = random_split(pair_dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader

def train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer, loss_fn, device, adv_weight=1e-3):
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0
    from models.FCNN3D import center_crop
    for x, y in tqdm(train_loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        # 训练判别器
        d_optimizer.zero_grad()
        fake = generator(x)
        fake_crop = center_crop(fake, (y.shape[-2], y.shape[-1]))
        real_out = discriminator(y)
        fake_out = discriminator(fake_crop.detach())
        d_loss = 0.5 * (torch.mean((real_out - 1) ** 2) + torch.mean(fake_out ** 2))  # LSGAN
        d_loss.backward()
        d_optimizer.step()
        # 训练生成器
        g_optimizer.zero_grad()
        fake = generator(x)
        fake_crop = center_crop(fake, (y.shape[-2], y.shape[-1]))
        fake_out = discriminator(fake_crop)
        rec_loss = loss_fn(fake_crop, y)
        adv_loss = torch.mean((fake_out - 1) ** 2)
        g_loss = rec_loss + adv_weight * adv_loss
        g_loss.backward()
        g_optimizer.step()
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
    return total_g_loss / len(train_loader), total_d_loss / len(train_loader)

def test_gan(generator, test_loader, device):
    generator.eval()
    metrics_list = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Test', leave=False):
            x, y = x.to(device), y.to(device)
            fake = generator(x)
            fake_np = fake.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()
            metrics = evaluate_metrics(fake_np, y_np, ratio=4)
            metrics_list.append(metrics)
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, required=True, help='指定patch数据的根目录或具体子目录，如datasets/patches/Chikusei 或 datasets/patches/Chikusei/X')
    args = parser.parse_args()
    patch_dir = args.patch_dir

    batch_size = 8
    lr = 1e-4
    epochs = 20
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available, using CPU.")

    train_loader, test_loader = get_dataloaders(patch_dir, batch_size=batch_size)

    generator = HSIGenerator(in_ch=1).to(device)
    discriminator = HSIPatchDiscriminator3D(in_ch=1).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs+1):
        g_loss, d_loss = train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer, loss_fn, device)
        metrics = test_gan(generator, test_loader, device)
        print(f"Epoch {epoch}: G_Loss={g_loss:.4f}, D_Loss={d_loss:.4f}, " +
              ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

    os.makedirs('./saved', exist_ok=True)
    torch.save(generator.state_dict(), "./saved/gan_generator_final.pth")
    torch.save(discriminator.state_dict(), "./saved/gan_discriminator_final.pth")
