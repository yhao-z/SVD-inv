from utils import generate_mask
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage import io as skio
import torch

masktype = 'uds_0.5'

class myDataset(Dataset):
    def __init__(self, data_list, length):
        self.data_list = data_list
        self.len = length

    def __getitem__(self, index):
        f = self.data_list[index]
        label = skio.imread(f)
        label = torch.from_numpy(label.astype(np.float32)).cuda()
        label = label.permute(2, 0, 1)
        return label

    def __len__(self):
        return self.len
    
test_list = sorted(glob.glob(os.path.join('./test_images','*.tiff')))
dataset_test = DataLoader(myDataset(test_list, len(test_list)), batch_size=1, shuffle=False)

MASK = []
for step, sample in enumerate(dataset_test):
    label = sample

    # generate under-sampling mask (random)
    nb, nc, nx, ny = label.shape
    mask = generate_mask([nx, ny, nc], float(masktype.split('_', 1)[1]), masktype.split('_', 1)[0])
    mask = np.transpose(mask, (2, 0, 1))
    mask = np.float32(mask)
    MASK.append(mask)
np.savez('./test_images/test_'+masktype, *MASK)