# import
import argparse
import os
import sys
import time
import glob
from datetime import datetime
import numpy as np
import scipy.io as scio
import shutil
from loguru import logger
from skimage import io as skio
from skimage.metrics import structural_similarity
from natsort import natsorted
# torch package
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms
# inner import
from models import TLR_Net
from utils import calc_SNR, calc_PSNR, ifft2c_mri, generate_mask


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
    

class Solver(object):
    def __init__(self, args):
        self.start_epoch = args.start_epoch
        self.end_epoch = args.end_epoch
        
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.niter = args.niter
        self.masktype = args.masktype
        
        self.svdtype = args.svdtype
        
        self.ModelName = args.ModelName
        self.weight = args.weight
        
        self.debug = args.debug
        # specify network
        self.net = TLR_Net(self.niter, self.svdtype).cuda()
        self.param_num = 0 # initialize it 0, later calc it
        
        self.archive()

        cifar_train = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=transforms.ToTensor(), download=True)
        cifar_test = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=transforms.ToTensor(), download=True) 
        cifar_dataset = ConcatDataset([cifar_train, cifar_test])
        self.dataset_train = DataLoader(cifar_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_list = sorted(glob.glob(os.path.join('./test_images','*.tiff')))
        self.dataset_test = DataLoader(myDataset(test_list, len(test_list)), batch_size=1, shuffle=False)
        logger.info('dataset loaded.')
        
        
    def train(self):
        # # load pre-weight
        # if self.weight is not None:
        #     logger.info('load weights.')
        #     # torch.save(model.state_dict(), './out/cnn_best_'+str(dnum)+'_'+masktype+'.pth')
        #     self.net.load_state_dict(torch.load(self.weight))
        
        # define lr and optimizer
        learning_rate = self.learning_rate
        learning_rate_decay = 0.95
        learning_rate = learning_rate * learning_rate_decay ** (self.start_epoch - 1)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = learning_rate_decay)

        # Iterate over epochs.
        total_step = 0
        loss = 0
        
        if self.weight is not None:
            logger.info('load weights.')
            self.net.load_state_dict(torch.load(self.weight))

        for epoch in range(self.start_epoch, self.end_epoch+1):
            for step, sample in enumerate(self.dataset_train):                
                # forward
                t0 = time.time()
                
                label, _ = sample

                if label.shape[0] < self.batch_size:
                    continue

                # generate under-sampling mask (random)\
                label = label.cuda()
                nb, nc, nx, ny = label.shape
                mask = generate_mask([nx, ny, 3], float(self.masktype.split('_', 1)[1]), self.masktype.split('_', 1)[0])
                mask = np.transpose(mask, (2, 0, 1))
                mask = torch.from_numpy(np.float32(mask)).cuda()

                # generate the undersampled data uds
                uds = label * mask

                # feed the data
                recon, svd_flags = self.net(uds, mask)
                recon_abs = torch.abs(recon)
                psnr = calc_PSNR(recon, label)

                # compute loss
                loss = loss_func(recon, label)
                                    
                # sum all the losses and avarage them to write the summary when epoch ends
                psnr_epoch = psnr_epoch + psnr if step != 0 else psnr
                loss_epoch = loss_epoch + loss.item() if step != 0 else loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if not loss.isfinite():
                    logger.error('gradient nan, training failed')
                    raise OverflowError('gradient nan, training failed')
                
                # if 1 not in svd_flags:
                #     try:
                #         loss.backward()
                #         nn.utils.clip_grad_norm_(self.net.parameters(), 100, norm_type='inf', error_if_nonfinite=True)
                #         optimizer.step()
                #     except:
                #         logger.warning('gradient nan')
                #         continue
                
                if self.param_num == 0:
                    self.param_num = np.sum([np.prod(v.shape) for v in self.net.parameters() if v.requires_grad])

                # log output
                if step % 100 == 0: 
                    logger.info('Epoch %d/%d, Step %d, Loss=%.3e, PSNR=%.2f, time=%.2f, lr=%.4e' % (epoch, self.end_epoch, step, loss.item(), psnr, (time.time() - t0), learning_rate))
                
                # record loss of each step
                self.train_writer.add_scalar('step/loss', loss, global_step=total_step)
                self.train_writer.add_scalar('step/PSNR', psnr, global_step=total_step)
                
                total_step += 1
                
            # At the end of epoch, print one message
            logger.info('Epoch %d/%d, Step %d, Loss=%.3e, PSNR=%.2f, time=%.2f, lr=%.4e' % (epoch, self.end_epoch, step, loss.item(), psnr, (time.time() - t0), learning_rate))
            
            # learning rate decay for each epoch
            scheduler.step()
            
            # record loss
            self.train_writer.add_scalar('epoch/loss', loss_epoch/(step+1), global_step=epoch)
            self.train_writer.add_scalar('epoch/PSNR', psnr_epoch/(step+1), global_step=epoch)
                
            # save model
            # save the latest epoch weights for continued training
            torch.save(self.net.state_dict(), self.weightdir+'/weight-latest'+'.pth')
            # every 10 epoches, we save the weights
            if epoch % 10 == 0 or epoch == 1:
                torch.save(self.net.state_dict(), self.weightdir+'/weight-'+str(epoch)+'.pth')

        self.test()


    def test(self):
        if self.weight is not None:
            logger.info('loading weights...')
            self.net.load_state_dict(torch.load(self.weight))
        logger.info('net initialized, testing...')
        SNRs = []
        PSNRs = []
        MSEs = []
        SSIMs = []
        masks = np.load('./test_images/test_'+self.masktype+'.npz')
        for step, sample in enumerate(self.dataset_test):
            label = sample

            # generate under-sampling mask (fix for test)
            mask = masks[masks.files[step]]
            mask = torch.from_numpy(np.float32(mask)).cuda()

            # generate the undersampled data k0
            uds = label * mask
            
            # feed the data
            t0 = time.time()
            recon, _ = self.net(uds, mask)
            t = time.time() - t0
            
            # if step == 8:
            #     scio.savemat(self.ModelName+'.mat', {'recon': recon.numpy()})
            #     scio.savemat('us.mat', {'us': ifft2c_mri(k0).numpy()})
            
            # calc the metrics
            SNR_ = calc_SNR(recon, label)
            PSNR_ = calc_PSNR(recon, label)
            MSE_ = loss_func(recon, label).item()

            SSIM_ = structural_similarity(recon.abs().detach().cpu().numpy()[0], label.abs().detach().cpu().numpy()[0], channel_axis=0, data_range=1.0)
            SNRs.append(SNR_)
            PSNRs.append(PSNR_)
            MSEs.append(MSE_)
            SSIMs.append(SSIM_)
            logger.info('data %d --> SER = \%.3f\, PSNR = \%.3f\, SSIM = \%.3f\, MSE = {%.3e}, t = %.2f' % (step, SNR_, PSNR_, SSIM_, MSE_, t))
            
        SNRs = np.array(SNRs)
        PSNRs = np.array(PSNRs)
        MSEs = np.array(MSEs)
        logger.info('SER = %.3f(%.3f), PSNR = %.3f(%.3f), SSIM = %.3f(%.3f), MSE = %.3e(%.3e)' % (np.mean(SNRs), np.std(SNRs), np.mean(PSNRs), np.std(PSNRs), np.mean(SSIMs), np.std(SSIMs), np.mean(MSEs), np.std(MSEs)))

    def archive(self):
        if not self.debug:
            # give the log dir and the model dir
            name_seq = [str(self.ModelName), str(self.masktype), str(self.svdtype), str(self.niter), str(self.batch_size), str(self.learning_rate)]
            model_id = '-'.join([name_seq[i] for i in [0,1,2]]) # it can be chosen flexiably
            TIMESTAMP = "{0:%Y%m%dT%H%M%S}".format(datetime.now())
            
            os.makedirs('./archive') if not os.path.exists('./archive') else None
            target =  './archive/' + model_id + '-' + TIMESTAMP
            os.makedirs(target) if not os.path.exists(target) else None
            
            # log
            train_logdir = target+'/logs/train'
            self.train_writer = SummaryWriter(train_logdir)
            # print logger
            # logger.remove(handler_id=None) # 清除之前的设置
            logger.add(sink=target+'/log.log', level='INFO') 
                
            # model
            self.weightdir = target+'/weights/'
            os.makedirs(self.weightdir) if not os.path.exists(self.weightdir) else None

            # adding exception handling
            try:
                shutil.copy('./main.py', target)
                shutil.copy('./Solver.py', target)
                shutil.copytree('./models', target+'/models')
            except IOError as e:
                logger.error("Unable to copy file. %s" % e)
            except:
                logger.error("Unexpected error:", sys.exc_info())
        elif self.debug:
            logger.remove()
            logger.add(sink = sys.stderr, level='DEBUG')
            # log
            train_logdir = './logs/train'
            self.train_writer = SummaryWriter(train_logdir)
            # model
            self.weightdir = './weights/'
            os.makedirs(self.weightdir) if not os.path.exists(self.weightdir) else None
            

def loss_func(y, y_):
    loss = torch.mean(torch.square((y - y_).abs()))

    return loss
