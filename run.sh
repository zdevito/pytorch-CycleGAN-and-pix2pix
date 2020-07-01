#!/usr/bin/env python
import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='eval something')
opts = parser.parse_args()
if opts.eval: 
    raise NotImplementedError()
else:
    train_args = "--dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan --display_id 0 --n_epochs 3 --n_epochs_decay 3"
    train(train_args.split(' '))

