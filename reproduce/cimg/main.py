import argparse
import os
import sys
from Solver import Solver
from loguru import logger
from utils import setup_seed

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

if __name__ == "__main__":
    
    setup_seed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU No.')
    
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch, begin with 1')
    parser.add_argument('--end_epoch', type=int, default=50, help='end epoch or number of epochs')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')

    parser.add_argument('--niter', type=int, default=10, help='number of network iterations')
    parser.add_argument('--masktype', type=str, default='uds_0.4')
    
    parser.add_argument('--svdtype', type=str, default='tf', help='orig, tf, clip, taylor, mine')

    parser.add_argument('--ModelName', type=str, default='TLR_Net', help='')

    parser.add_argument('--weight', type=str, default=None)
    
    parser.add_argument('--debug', type=int, default=0, help='1:debug, 0 running')

    args = parser.parse_args()
    
        
    solver = Solver(args)
    logger.info(args)
    logger.critical('All done! Running now ------')
    
    solver.train()
