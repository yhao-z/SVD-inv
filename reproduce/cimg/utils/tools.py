import numpy as np
import torch
import random
import os
 
def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False

def calc_SNR(y, y_):
    y = y.detach().cpu().numpy()
    y_ = y_.detach().cpu().numpy()
    y = np.array(y).flatten()
    y_ = np.array(y_).flatten()
    err = np.linalg.norm(y_ - y) ** 2
    snr = 10 * np.log10(np.linalg.norm(y_) ** 2 / err)

    return snr


def calc_PSNR(recon, label):
    recon = recon.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    recon = np.array(recon).flatten()
    label = np.array(label).flatten()
    err = np.linalg.norm(label - recon) ** 2
    max_label = np.max(np.abs(label))
    N = np.prod(recon.shape)
    psnr = 10 * np.log10(N * max_label ** 2 / err)

    return psnr


def fft2c_mri(x):
    # nb nt nx ny
    X = torch.fft.ifftshift(x, dim=(2,3))
    X = torch.fft.fft2(X)
    X = torch.fft.fftshift(X, dim=(2,3))
    nx, ny = torch.tensor(X.shape[2:])
    X = torch.div(X, torch.sqrt(nx * ny))
    return X


def ifft2c_mri(X):
    # nb nt nx ny
    x = torch.fft.ifftshift(X, dim=(2,3))
    x = torch.fft.ifft2(x)
    x = torch.fft.fftshift(x, dim=(2,3))
    nx, ny = torch.tensor(x.shape[2:])
    x = torch.mul(x, torch.sqrt(nx * ny))
    return x