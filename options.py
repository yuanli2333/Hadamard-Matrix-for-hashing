import argparse

parser = argparse.ArgumentParser(description='CHV')

# model
parser.add_argument('--model_type', type=str, default='Resnet', help='The type of base model')

# Training
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='training epoch')
#parser.add_argument('--gpu_ids', nargs='+', type=int, default=None, help='gpu devices ids')
parser.add_argument('--gpus', type=str, default="0,1", help="define gpu id")
parser.add_argument('--num_iteration', type=int, default=60, help='number of iteration')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size for training')  # batch_size most be even in this project
parser.add_argument('--eval_frequency', type=int, default=4, help='the evaluate frequency for testing')
parser.add_argument('--data_name', type=str, default='imagenet', help='imagenet or coco...')
parser.add_argument('--num_class', type=int, default=100, help='The number of classes')
parser.add_argument('--workers', type=int, default=8, help='number of data loader workers.')
parser.add_argument('--multi_lr', type=float, default=0.01, help= 'multiplier for learning rate')
parser.add_argument('--lambda0', type=float, default=1, help='hyper-parameters 0')
parser.add_argument('--lambda1', type=float, default=0.2, help='hyper-parameters 1')
parser.add_argument('--lambda2', type=float, default=0.05, help='hyper-parameters 1')

# Hashing
parser.add_argument('--hash_bit', type=int, default='64', help = 'hash bit,it can be 8, 16, 32, 64, 128...')
parser.add_argument('--batch_size_hash', type=int, default=40, help='the batch size for training')  # batch_size most be even in this project

# Testing
parser.add_argument('--R', type=int, default=1000, help='MAP@R')
parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
parser.add_argument('--model_name', type=str, default='imagenet_64bit_0.873_resnet50.pkl', help='Put any model you want to test here')
