# torch package
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from utils import fft2c_mri, ifft2c_mri
from loguru import logger
from .mysvd import *

class CONV_OP(nn.Module):
    def __init__(self, n_in=16, n_out=16, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module('conv', nn.Conv2d(n_in, n_out, 3, stride=1, padding='same', bias=False))
        if ifactivate == True:
            self.seq.add_module('relu', nn.ReLU())

    def forward(self, input):
        res = self.seq(input)
        return res


class TLRCell(nn.Module):
    def __init__(self, svdtype, is_last=False):
        super(TLRCell, self).__init__()
        
        self.svdtype = svdtype

        if is_last:
            self.thres_coef = nn.Parameter(torch.Tensor([-2]))
            self.eta = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        else:
            self.thres_coef = nn.Parameter(torch.Tensor([-2]))
            self.eta = nn.Parameter(torch.Tensor([1]))

        self.conv_1 = CONV_OP(n_in=3, n_out=16, ifactivate=True)
        self.conv_2 = CONV_OP(n_in=16, n_out=16, ifactivate=True)
        self.conv_3 = CONV_OP(n_in=16, n_out=3, ifactivate=False)

        self.conv_4 = CONV_OP(n_in=3, n_out=16, ifactivate=True)
        self.conv_5 = CONV_OP(n_in=16, n_out=16, ifactivate=True)
        self.conv_6 = CONV_OP(n_in=16, n_out=3, ifactivate=False)

    def forward(self, data):
        x_rec, L, A, d, mask = data

        A, svd_flag = self.lowrank_step(x_rec + L)
        x_rec = self.x_step(L, A, d, mask)
        L = self.L_step(L, x_rec, A)

        data[0] = x_rec
        data[1] = L
        data[2] = A

        return data, svd_flag

    def x_step(self, L, A, d, mask):
        x_rec = (1-mask) * (A - L) + mask * d
        return x_rec

    def lowrank_step(self, x):
        svd_flag = 0
        x_in = x
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_3_c = x_3

        # try:
        assert x_3_c.isfinite().all()
        
        if self.svdtype == 'orig':
            Ut, St, Vth = torch.linalg.svd(x_3_c, full_matrices=False)
            thres = torch.sigmoid(self.thres_coef) * St[..., 0] # [batch, Nc, Nx, Ny]
            thres = torch.unsqueeze(thres, -1)
            St = F.relu(St - thres)

            St = St.type_as(Ut)
            x_soft = Ut @ torch.diag_embed(St) @ Vth
        elif self.svdtype == 'tf':
            Ut, St, Vt = svd_tf.apply(x_3_c)
            St = St.diagonal(dim1=-2, dim2=-1)
            thres = torch.sigmoid(self.thres_coef) * St[..., 0] # [batch, Nc, Nx, Ny]
            thres = torch.unsqueeze(thres, -1)
            St = F.relu(St - thres)

            St = St.type_as(Ut)
            x_soft = Ut @ torch.diag_embed(St) @ Vt.mH
        elif self.svdtype == 'clip':
            Ut, St, Vt = svd_clip.apply(x_3_c)
            St = St.diagonal(dim1=-2, dim2=-1)
            thres = torch.sigmoid(self.thres_coef) * St[..., 0] # [batch, Nc, Nx, Ny]
            thres = torch.unsqueeze(thres, -1)
            St = F.relu(St - thres)

            St = St.type_as(Ut)
            x_soft = Ut @ torch.diag_embed(St) @ Vt.mH
        elif self.svdtype == 'taylor':
            Ut, St, Vt = svd_taylor.apply(x_3_c)
            St = St.diagonal(dim1=-2, dim2=-1)
            thres = torch.sigmoid(self.thres_coef) * St[..., 0] # [batch, Nc, Nx, Ny]
            thres = torch.unsqueeze(thres, -1)
            St = F.relu(St - thres)

            St = St.type_as(Ut)
            x_soft = Ut @ torch.diag_embed(St) @ Vt.mH
        elif self.svdtype == 'mine':
            Ut, St, Vt = svd_inv.apply(x_3_c)
            St = St.diagonal(dim1=-2, dim2=-1)
            thres = torch.sigmoid(self.thres_coef) * St[..., 0] + 1e-10 # [batch, Nc, Nx, Ny]
            thres = torch.unsqueeze(thres, -1)
            St = F.relu(St - thres)

            St = St.type_as(Ut)
            x_soft = Ut @ torch.diag_embed(St) @ Vt.mH
        else:
            raise NotImplemeNcedError
        # except:
        #     x_soft = x_3_c
        #     logger.warning('svd failed')
        #     svd_flag = 1

        x_soft = x_soft.type(torch.float32)
        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        A = x_6 + x_in

        return A, svd_flag

    def L_step(self, L, x_rec, A):
        eta = F.relu(self.eta)
        return L + torch.mul(x_rec - A, eta)


class TLR_Net(nn.Module):
    def __init__(self, niter, svdtype):
        super(TLR_Net, self).__init__()
        self.niter = niter
        self.svdtype = svdtype
        self.celllist = []

        for i in range(self.niter - 1):
            self.celllist.append(TLRCell(self.svdtype))
        self.celllist.append(TLRCell(self.svdtype, is_last=True))
        
        self.fcs = nn.ModuleList(self.celllist)

    def forward(self, d, mask):
        """
        d: undersampled k-space
        mask: sampling mask
        """
        # nb, nc, Nc, nx, ny = d.shape
        x_rec = d
        A = torch.zeros_like(x_rec)
        L = torch.zeros_like(x_rec)
        data = [x_rec, L, A, d, mask]

        svd_flags = []
        for i in range(self.niter):
            data, svd_flag = self.fcs[i](data)
            svd_flags.append(svd_flag)

        x_rec = data[0]

        return x_rec, svd_flags

