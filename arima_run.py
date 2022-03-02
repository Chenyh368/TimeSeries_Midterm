import random
import numpy as np
import argparse
from arima_utils.ARIMA import ExpArima

fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ARIMA model')

#======================= Basic Config ========================


#======================= ARIMA ========================
parser.add_argument('--d', type=int, default=1)
parser.add_argument('--p', type=int, default=1)
parser.add_argument('--q', type=int, default=1)

#======================= Data ========================
parser.add_argument('--root_path', type=str, default='~/data/informer_dataset/midterm', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='CleanBeijing.csv', help='data file')
parser.add_argument('--split', type=float, default=0.5)
parser.add_argument('--target', type=str, default='TEMP', help='target feature')
parser.add_argument('--scaler', type=bool, default=False)


args = parser.parse_args()
arima = ExpArima(args)