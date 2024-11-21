import torch
import numpy as np

def geometric_approximation(s):
    # this is the Taylor approximation of the function
    # no risk of duplicate singular values
    s = torch.square(s)
    s = torch.where(s<1e-30, 0, s)
    I = torch.eye(s.shape[-1]).type(s.dtype).to(s.device)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p.isfinite(), p, 0)
    p = torch.where(p < 1., p, 1. / p)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones_like(a1).to(s.device) - I
    a1 = torch.where(a1.isfinite(), a1, 0)
    p_app = torch.ones_like(p).to(s.device)
    p_hat = torch.ones_like(p).to(s.device)
    for i in range(9):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


def F_tf(s):
    # this is the tensorflow implementation of the function
    # it just replace the NaN values with 0
    s = torch.square(s)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    a1 = torch.where((a1-a1_t)==0, 0, 1./(a1 -a1_t))
    a1 = torch.where(a1.isfinite(), a1, 0)
    return a1


def F_orig(s):
    # this is the original function
    # it has risk of duplicate singular values
    s = torch.square(s)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    I = torch.eye(a1.shape[-1]).type(a1.dtype).to(s.device)
    a1 = torch.where(I==1, 0, 1./(a1 -a1_t))
    return a1

def F_clip(s):
    # this is the original function
    # it has risk of duplicate singular values
    s = torch.square(s)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    I = torch.eye(a1.shape[-1]).type(a1.dtype).to(s.device)
    F = torch.where(I==1, 0, 1./(a1 -a1_t))
    F = torch.where((a1==0) & (a1_t==0), 0, F)
    F = torch.where(F.isfinite(), F, 1e30)
    return F


def F_and_T(s):
    s = torch.square(s)
    s = torch.where(s<1e-30, 0, s)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    I = torch.eye(a1.shape[-1]).type(a1.dtype).to(s.device)#.cuda()
    a = torch.where((I==1) | ((a1==0) & (a1_t==0)), 0, 1./(a1 -a1_t))
    F = torch.where(a.isfinite(), a, 0)
    logi = F.abs()>1e30
    F = torch.where(logi, 0, F)
    T = torch.where(a.abs().isinf(), 1.0/(a1.sqrt()), 0) # 奇异值相等的部分，归类到T里面计算，因为在F中为inf并且置为0了
    T = torch.where(logi, 1.0/(a1.sqrt()), T)
    return F, T
    

class svd_taylor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        s = torch.diag_embed(s)
        ctx.save_for_backward(x, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds, dl_dv):
        x, u, s, v = ctx.saved_tensors
        
        s_inv = torch.where(s>0., torch.diag_embed(1.0 / s.diagonal(dim1=-2, dim2=-1)), 0)
        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        ################### the only diff #######################
        F = geometric_approximation(s.diagonal(dim1=-2, dim2=-1))
        Fmat_u = F.type(utdu.dtype) * (utdu-utdu.mH)
        Fmat_v = F.type(utdu.dtype) * (vtdv-vtdv.mH)
        #########################################################
        c_u1 = Fmat_u @ s.type(u.dtype)
        c_u1 = u @ c_u1
        Im = torch.eye(u.shape[-2]).type(u.dtype).to(u.device)
        c_u2 = Im - u @ u.mH
        c_u2 = c_u2 @ dl_du @ s_inv.type(u.dtype)
        c_u = (c_u1 + c_u2) @ v.mH
        Ik = torch.eye(s.shape[-1]).type(s.dtype).to(u.device)
        c_s = u @ (Ik*dl_ds.type(u.dtype)) @ v.mH
        c_v1 = s.type(u.dtype) @ Fmat_v @ v.mH
        In = torch.eye(v.shape[-2]).type(v.dtype).to(u.device)
        c_v2 = In - v @ v.mH
        c_v2 = s_inv.type(u.dtype) @ dl_dv.mH @ c_v2
        c_v = u @ (c_v1 + c_v2)
        dl_dx = c_u + c_s + c_v
        return dl_dx


class svd_tf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        s = torch.diag_embed(s)
        ctx.save_for_backward(x, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds, dl_dv):
        x, u, s, v = ctx.saved_tensors
        
        s_inv = torch.where(s>0., torch.diag_embed(1.0 / s.diagonal(dim1=-2, dim2=-1)), 0)
        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        ################### the only diff #######################
        F = F_tf(s.diagonal(dim1=-2, dim2=-1))
        Fmat_u = F.type(utdu.dtype) * (utdu-utdu.mH)
        Fmat_v = F.type(utdu.dtype) * (vtdv-vtdv.mH)
        #########################################################
        c_u1 = Fmat_u @ s.type(u.dtype)
        c_u1 = u @ c_u1
        Im = torch.eye(u.shape[-2]).type(u.dtype).to(u.device)
        c_u2 = Im - u @ u.mH
        c_u2 = c_u2 @ dl_du @ s_inv.type(u.dtype)
        c_u = (c_u1 + c_u2) @ v.mH
        Ik = torch.eye(s.shape[-1]).type(s.dtype).to(u.device)
        c_s = u @ (Ik*dl_ds.type(u.dtype)) @ v.mH
        c_v1 = s.type(u.dtype) @ Fmat_v @ v.mH
        In = torch.eye(v.shape[-2]).type(v.dtype).to(u.device)
        c_v2 = In - v @ v.mH
        c_v2 = s_inv.type(u.dtype) @ dl_dv.mH @ c_v2
        c_v = u @ (c_v1 + c_v2)
        dl_dx = c_u + c_s + c_v
        assert dl_dx.isfinite().all() # check if there is nan
        return dl_dx


class svd_orig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        s = torch.diag_embed(s)
        ctx.save_for_backward(x, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds, dl_dv):
        x, u, s, v = ctx.saved_tensors
        
        s_inv = torch.where(s>0., torch.diag_embed(1.0 / s.diagonal(dim1=-2, dim2=-1)), 0)
        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        ################### the only diff #######################
        F = F_orig(s.diagonal(dim1=-2, dim2=-1))
        Fmat_u = F.type(utdu.dtype) * (utdu-utdu.mH)
        Fmat_v = F.type(utdu.dtype) * (vtdv-vtdv.mH)
        #########################################################
        c_u1 = Fmat_u @ s.type(u.dtype)
        c_u1 = u @ c_u1
        Im = torch.eye(u.shape[-2]).type(u.dtype).to(u.device)
        c_u2 = Im - u @ u.mH
        c_u2 = c_u2 @ dl_du @ s_inv.type(u.dtype)
        c_u = (c_u1 + c_u2) @ v.mH
        Ik = torch.eye(s.shape[-1]).type(s.dtype).to(u.device)
        c_s = u @ (Ik*dl_ds.type(u.dtype)) @ v.mH
        c_v1 = s.type(u.dtype) @ Fmat_v @ v.mH
        In = torch.eye(v.shape[-2]).type(v.dtype).to(u.device)
        c_v2 = In - v @ v.mH
        c_v2 = s_inv.type(u.dtype) @ dl_dv.mH @ c_v2
        c_v = u @ (c_v1 + c_v2)
        dl_dx = c_u + c_s + c_v
        return dl_dx
    
    
class svd_clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        s = torch.diag_embed(s)
        ctx.save_for_backward(x, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds, dl_dv):
        x, u, s, v = ctx.saved_tensors
        
        s_inv = torch.where(s>0., torch.diag_embed(1.0 / s.diagonal(dim1=-2, dim2=-1)), 0)
        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        ################### the only diff #######################
        F = F_clip(s.diagonal(dim1=-2, dim2=-1))
        Fmat_u = F.type(utdu.dtype) * (utdu-utdu.mH)
        Fmat_v = F.type(utdu.dtype) * (vtdv-vtdv.mH)
        #########################################################
        c_u1 = Fmat_u @ s.type(u.dtype)
        c_u1 = u @ c_u1
        Im = torch.eye(u.shape[-2]).type(u.dtype).to(u.device)
        c_u2 = Im - u @ u.mH
        c_u2 = c_u2 @ dl_du @ s_inv.type(u.dtype)
        c_u = (c_u1 + c_u2) @ v.mH
        Ik = torch.eye(s.shape[-1]).type(s.dtype).to(u.device)
        c_s = u @ (Ik*dl_ds.type(u.dtype)) @ v.mH
        c_v1 = s.type(u.dtype) @ Fmat_v @ v.mH
        In = torch.eye(v.shape[-2]).type(v.dtype).to(u.device)
        c_v2 = In - v @ v.mH
        c_v2 = s_inv.type(u.dtype) @ dl_dv.mH @ c_v2
        c_v = u @ (c_v1 + c_v2)
        dl_dx = c_u + c_s + c_v
        return dl_dx
    

class svd_inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        s = torch.diag_embed(s)
        ctx.save_for_backward(x, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds, dl_dv):
        x, u, s, v = ctx.saved_tensors
        
        s_inv = torch.where(s>0., torch.diag_embed(1.0 / s.diagonal(dim1=-2, dim2=-1)), 0)
        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        ################### the only diff #######################
        F, T = F_and_T( s.diagonal(dim1=-2, dim2=-1) )    
        Fmat_u = F.type(utdu.dtype) * (utdu-utdu.mH)
        Fmat_v = F.type(utdu.dtype) * (vtdv-vtdv.mH)
        c_u1 = Fmat_u @ s.type(u.dtype) + T.type(u.dtype) * utdu
        #########################################################
        c_u1 = u @ c_u1
        Im = torch.eye(u.shape[-2]).type(u.dtype).to(u.device)
        c_u2 = Im - u @ u.mH
        c_u2 = c_u2 @ dl_du @ s_inv.type(u.dtype)
        c_u = (c_u1 + c_u2) @ v.mH
        Ik = torch.eye(s.shape[-1]).type(s.dtype).to(u.device)
        c_s = u @ (Ik*dl_ds.type(u.dtype)) @ v.mH
        c_v1 = s.type(u.dtype) @ Fmat_v @ v.mH
        In = torch.eye(v.shape[-2]).type(v.dtype).to(u.device)
        c_v2 = In - v @ v.mH
        c_v2 = s_inv.type(u.dtype) @ dl_dv.mH @ c_v2
        c_v = u @ (c_v1 + c_v2)
        dl_dx = c_u + c_s + c_v
        assert dl_dx.isfinite().all() # check if there is nan
        assert (dl_dx.abs()<1e16).all() # check if there is large value
        return dl_dx