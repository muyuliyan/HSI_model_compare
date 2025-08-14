# detector.py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(pred, target, data_range=1.0):
    """计算PSNR，输入为numpy数组，shape: [C, H, W]"""
    return psnr(target, pred, data_range=data_range)

def compute_ssim(pred, target, data_range=1.0):
    """计算平均SSIM，逐波段求平均"""
    C = pred.shape[0]
    ssim_total = 0
    for i in range(C):
        ssim_total += ssim(target[i], pred[i], data_range=data_range)
    return ssim_total / C

def compute_rmse(pred, target):
    """均方根误差"""
    return np.sqrt(np.mean((pred - target) ** 2))

def compute_sam(pred, target):
    """光谱角度映射（SAM），单位：度"""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    dot = np.sum(pred_flat * target_flat, axis=0)
    norm_pred = np.linalg.norm(pred_flat, axis=0)
    norm_target = np.linalg.norm(target_flat, axis=0)
    cos_theta = dot / (norm_pred * norm_target + 1e-8)
    cos_theta = np.clip(cos_theta, -1, 1)
    sam = np.arccos(cos_theta)
    return np.mean(np.degrees(sam))

def compute_ergas(pred, target, ratio=4):
    """ERGAS指标，ratio为空间分辨率比（如4倍超分则为4）"""
    # patch 已归一化到[0,1]，直接用归一化值计算ERGAS
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    C = pred.shape[0]
    ergas_sum = 0
    for i in range(C):
        rmse_band = np.sqrt(np.mean((pred[i] - target[i]) ** 2))
        mean_band = np.mean(target[i])
        ergas_sum += (rmse_band / (mean_band + 1e-8)) ** 2
    ergas = 100 / ratio * np.sqrt(ergas_sum / C)
    return ergas

def evaluate_metrics(pred, target, ratio=4):
    """汇总所有指标"""
    return {
        "PSNR": compute_psnr(pred, target),
        "SSIM": compute_ssim(pred, target),
        "SAM": compute_sam(pred, target),
        "ERGAS": compute_ergas(pred, target, ratio),
        "RMSE": compute_rmse(pred, target)
    }