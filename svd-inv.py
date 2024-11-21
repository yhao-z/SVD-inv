import torch
import numpy as np
from svds import svd_orig, svd_tf, svd_clip, svd_taylor


def F_and_T_inv(s):
    # this is our svd-inv's function
    # no risk of duplicate singular values
    s = torch.square(s)
    s = torch.where(s<1e-30, 0, s)
    a1 = torch.tile(s.unsqueeze(-1),(s.shape[-1],)).mT
    a1_t = a1.mT
    I = torch.eye(a1.shape[-1]).type(a1.dtype).to(s.device)
    a = torch.where((I==1) | ((a1==0) & (a1_t==0)), 0, 1./(a1 -a1_t))
    F = torch.where(a.isfinite(), a, 0)
    logi = F.abs()>1e30
    F = torch.where(logi, 0, F)
    T = torch.where(a.abs().isinf(), 1.0/(a1.sqrt()), 0) # The parts with equal singular values are classified into T
    T = torch.where(logi, 1.0/(a1.sqrt()), T)
    return F, T


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
        F, T = F_and_T_inv( s.diagonal(dim1=-2, dim2=-1) )    
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


if __name__ == '__main__':
    # test the implementation and compare with the tensorflow or orginal implementation using a simple example
    
    # two data for easy test
    matrix_normal = np.array([[ 5, 8, 6, 7, 2],
            [ 8, 5, 7, 4, 3],
            [ 6, 7, 5, 3, 8],
            [ 7, 4, 3, 5, 9]]) # normal one, no duplicate singular values
    matrix_dup = np.zeros((2,2,2))
    matrix_dup[0,:,:] = np.array([[ 2, 4 ], [ -4, 2 ]]) # has duplicate singular values: sqrt(20)
    matrix_dup[1,:,:] = np.array([[ 2, 4 ], [ -4, 2.0001 ]]) # has duplicate singular values: sqrt(20)
    
    # select a data
    # if matrix_normal is selected, all the svds should have similar or even same gradient with the pytorch official svd
    # if matrix_dup    is selected, pytorch official svd and svd_orig give nan gradient, while the others should give a valid gradient
    #                               the gradient for matrix_dup[0,:,:] and [1,:,:] should similar, since there is only a small difference
    #                               only our svd-inv gives similar gradient, demonstrating the accuracy.
    M0 = matrix_normal
    
    # pytorch official svd
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, vh = torch.linalg.svd(M, full_matrices=False)
    y = u @ (s+0.1).diag_embed() @ vh
    l = torch.sum(y)
    l.backward()
    print('pytorch svd')
    print(M.grad)
    g = M.grad.clone()
    print(torch.dist(M.grad, g))

    # we implement the svd by ourself using the mathematical derivation
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, v = svd_orig.apply(M)
    y = u @ (s.diagonal(dim1=-2, dim2=-1)+0.1).diag_embed() @ v.mH
    l = torch.sum(y)
    l.backward()
    print('svd_orig')
    print(M.grad)
    print(torch.dist(M.grad, g))
    
    # the tensorflow used svd
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, v = svd_tf.apply(M)
    y = u @ (s.diagonal(dim1=-2, dim2=-1)+0.1).diag_embed() @ v.mH
    l = torch.sum(y)
    l.backward()
    print('svd_tf')
    print(M.grad)
    # g = M.grad.clone()
    print(torch.dist(M.grad, g))
    
    # the svd with gradient clip (implemented by ourself)
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, v = svd_clip.apply(M)
    y = u @ (s.diagonal(dim1=-2, dim2=-1)+0.1).diag_embed() @ v.mH
    l = torch.sum(y)
    l.backward()
    print('svd_clip')
    print(M.grad)
    print(torch.dist(M.grad, g))
    
    # the svd using taylor approximation (code is refered to the svd-talor github repo https://github.com/WeiWangTrento/Robust-Differentiable-SVD)
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, v = svd_taylor.apply(M)
    y = u @ (s.diagonal(dim1=-2, dim2=-1)+0.1).diag_embed() @ v.mH
    l = torch.sum(y)
    l.backward()
    print('svd_taylor')
    print(M.grad)
    print(torch.dist(M.grad, g))

    # our svd inv
    M = torch.from_numpy(M0.astype(np.float32))
    M.requires_grad = True
    u, s, v = svd_inv.apply(M)
    y = u @ (s.diagonal(dim1=-2, dim2=-1)+0.1).diag_embed() @ v.mH
    l = torch.sum(y)
    l.backward()
    print('svd_inv')
    print(M.grad)
    print(torch.dist(M.grad, g))