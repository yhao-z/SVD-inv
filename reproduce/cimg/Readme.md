# SVD-inv Evaluation in Color Image Compressed Sensing

The network is actually a concise deep unrolling network that utilize the tensor low-rank property of the color image. The network structure is as follows. It consists three parts, X, Z, L. The Z part enforces the low rankness in the CNN-transformed domain. The singular value thresholding (SVT) is utilized to realize the low-rank property.

![image-20241121204610147](E:\Yhao\_markdown_pics\image-20241121204610147.png)

### Environment

- the essential requirements is listed in `requirements.txt`.  run `pip install -r requirements.txt` to install them.
- we just run the code in Windows 11 platform with NVIDIA Geforce 1080Ti GPU.
- we also test the code in ubuntu. it's ok to run.

### Training and Notes

- you could train the network by `python main.py`
- `testmask.py` is just used to generate the undersampling masks for the test images in `test_images` folder, e.g., `test_uds_0.1.npz`. you can just ignore it if you don't care the test undersampling mask.
- all other info can be found in our paper.