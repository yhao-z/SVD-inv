# SVD-inv

This is the official code of the paper, **Differentiable SVD based on Moore-Penrose Pseudoinverse for Inverse Imaging Problems**. 

- The main contribution essentially lies in the **mathematical derivation and code for a new and accurate differentiable SVD**. 
- The SVD-inv may not achieve obviously superior performance, if we only change the SVD function. After all, a supervised network may need hundreds of thousands training steps with a supervised loss. The SVD only encounters gradient problem in rare cases (there have two duplicated singular value in one matrix). One or two deviations won't matter much.
- BUT, it's very important to implement the SVD correctly from the perspective of mathematics.

### Usage

- You could just copy the `svd-inv.py` file into your project folder and import the `svd_inv` class. 

- use `U, S, V = svd_inv.apply(x)` to calculate the SVD of the input matrix `x`

- Note that, our `svd_inv` is not same as the PyTorch `torch.linalg.svd`, see the following example.

  ```shell
  ##### PyTorch SVD ##### 
  >>> A = torch.randn(5, 3)
  >>> U, S, Vh = torch.linalg.svd(A, full_matrices=False)
  >>> U.shape, S.shape, Vh.shape
  (torch.Size([5, 3]), torch.Size([3]), torch.Size([3, 3]))
  >>> torch.dist(A, U @ torch.diag(S) @ Vh)
  tensor(1.0486e-06)
  
  ##### OUR SVD-inv ##### 
  >>> A = torch.randn(5, 3)
  >>> U, S, V = svd_inv.apply(A) # here is V not Vh
  >>> U.shape, S.shape, V.shape # here S is a diagnal singular value matrix
  (torch.Size([5, 3]), torch.Size([3, 3]), torch.Size([3, 3]))
  >>> torch.dist(A, U @ S @ V.mH) # no need to diag(S), V needs conj&transpose
  tensor(1.0486e-06)
  ```

### Simply Compare SVD-inv with Other SVDs

- just run `python svd_inv.py`

- there have two matrix to evaluate the SVDs

  ```
      matrix_normal = np.array([[ 5, 8, 6, 7, 2],
              [ 8, 5, 7, 4, 3],
              [ 6, 7, 5, 3, 8],
              [ 7, 4, 3, 5, 9]]) # normal one, no duplicate singular values
      matrix_dup = np.zeros((2,2,2))
      matrix_dup[0,:,:] = np.array([[ 2, 4 ], [ -4, 2 ]]) # has duplicate singular values: sqrt(20)
      matrix_dup[1,:,:] = np.array([[ 2, 4 ], [ -4, 2.0001 ]]) 
  ```

- if **matrix_normal** is selected, all the svds should have similar or even same gradient with the pytorch official svd

- if **matrix_dup**        is selected, pytorch official svd and svd_orig give NaN gradient, while the others should give a valid gradient

  the gradient for `matrix_dup[0,:,:]` and `[1,:,:]` should similar, since there is only a small difference

  **only our svd-inv gives similar gradient, demonstrating the accuracy.**

### Reproduce the results in the paper

- The code for the application of *color image compressive sensing* is attached in `./reproduce/cimg` folder, see its [Readme](./reproduce/cimg/Readme.md) for more details.
- The code for MRI reconstruction is not attached, since the dataset is big and we think no need to upload it.
- The *color image compressive sensing* is enough to indicate the efficacy of SVD-inv.

### Reference

If this code is useful for your work, please cite our paper.

> @misc{zhang2024differentiablesvdbasedmoorepenrose,
>       title={Differentiable SVD based on Moore-Penrose Pseudoinverse for Inverse Imaging Problems}, 
>       author={Yinghao Zhang and Yue Hu},
>       year={2024},
>       eprint={2411.14141},
>       archivePrefix={arXiv},
>       primaryClass={math.NA},
>       url={https://arxiv.org/abs/2411.14141}, 
> }
