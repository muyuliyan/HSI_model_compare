# trainDEN.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from metrics.detector import evaluate_metrics
import argparse
from tqdm import tqdm

# 懒加载Dataset（与超分一致，假设patch为[band, H, W]，噪声已加好）
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
		patch = arr[i].copy()  # [C, H, W] noisy
		return patch

# 训练/测试集
def get_dataloaders(patch_dir, batch_size=8, test_ratio=0.1, model_name='MADNet3D'):
	dataset = PatchDataset(patch_dir)
	class PairDataset(torch.utils.data.Dataset):
		def __init__(self, base_dataset, model_name):
			self.base_dataset = base_dataset
			self.model_name = model_name
		def __len__(self):
			return len(self.base_dataset)
		def __getitem__(self, idx):
			patch = self.base_dataset[idx]
			# SDeCNN/HCANet 直接用 [B, C, H, W]，其他模型加 unsqueeze(0)
			if isinstance(patch, np.ndarray):
				if patch.ndim == 4 and patch.shape[0] == 2:
					if self.model_name in ['SDeCNN', 'HCANet']:
						return torch.from_numpy(patch[0]).float(), torch.from_numpy(patch[1]).float()
					else:
						return torch.from_numpy(patch[0]).float().unsqueeze(0), torch.from_numpy(patch[1]).float().unsqueeze(0)
				elif patch.ndim == 3:
					if self.model_name in ['SDeCNN', 'HCANet']:
						return torch.from_numpy(patch).float(), torch.from_numpy(patch).float()
					else:
						return torch.from_numpy(patch).float().unsqueeze(0), torch.from_numpy(patch).float().unsqueeze(0)
				else:
					raise ValueError("Patch shape不符，需为[2, C, H, W]或[C, H, W]")
			else:
				raise ValueError("Patch数据类型不符，需为numpy.ndarray")
	pair_dataset = PairDataset(dataset, model_name)
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
		loss = loss_fn(pred, y)
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
			pred_np = pred.squeeze().cpu().numpy()
			y_np = y.squeeze().cpu().numpy()
			metrics = evaluate_metrics(pred_np, y_np, ratio=1)  # 去噪ratio=1
			metrics_list.append(metrics)
	avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
	return avg_metrics

def import_model(model_name):
	if model_name == 'MADNet3D':
		from DENmodels.MADNet3D import MADNet3D
		return MADNet3D
	if model_name == 'Transformer':
		from DENmodels.Transformer import HSIDenoiseTransformer
		return HSIDenoiseTransformer
	if model_name == 'SDeCNN':
		from DENmodels.SDeCNN import HSI_SDeCNN
		return HSI_SDeCNN
	if model_name == 'HCANet':
		from DENmodels.HCANet import HCANet
		return HCANet
	else:
		raise ValueError('未知模型')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--patch_dir', type=str, required=True, help='指定patch数据的根目录或具体子目录，如datasets/patches/Chikusei 或 datasets/patches/Chikusei/X')
	parser.add_argument('--model', type=str, required=True, help='选择模型：')
	args = parser.parse_args()
	patch_dir = args.patch_dir

	batch_size = 2
	lr = 1e-3
	epochs = 20
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, test_loader = get_dataloaders(patch_dir, batch_size=batch_size, model_name=args.model)

	ModelClass = import_model(args.model)
	# 获取通道数（波段数）
	channels = next(iter(train_loader))[0].shape[1]
	if args.model == 'SDeCNN':
		model = ModelClass(num_bands=channels).to(device)
	elif args.model == 'HCANet':
		model = ModelClass(in_bands=channels).to(device)
	else:
		model = ModelClass(in_channels=1).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = torch.nn.MSELoss()

	for epoch in range(1, epochs+1):
		train_loss = train(model, train_loader, optimizer, loss_fn, device)
		metrics = test(model, test_loader, device)
		print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, " +
			  ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

	os.makedirs('./saved', exist_ok=True)
	torch.save(model.state_dict(), f"./saved/{args.model.lower()}_denoise_final.pth")
