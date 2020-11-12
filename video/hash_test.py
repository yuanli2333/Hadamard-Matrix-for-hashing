import numpy as np
import dataset
import os
import os
import json
import socket
import logging
import argparse

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_model import train_model
from network.symbol_builder import get_symbol
from data import iterator_factory


def mean_average_precision(database_hash, test_hash, database_labels, test_labels, args):  # R = 1000
    # binary the hash code
    R = args.R
    T = args.T
    database_hash[database_hash<T] = -1
    database_hash[database_hash>=T] = 1
    test_hash[test_hash<T] = -1
    test_hash[test_hash>=T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T) 
    ids = np.argsort(-sim, axis=0)  
    ids_100 = ids[:100, :]
    file_dir = 'dataset/' + args.dataset
    np.save(file_dir + '/ids.npy', ids_100)
    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i]  # the first test labels
        idx = ids[:, i]
        imatch = (database_labels[idx[0:R]] == label) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        #all_relevant = np.sum(database_labels == label, axis=1) > 0
        #all_num = np.sum(all_relevant)
        #r = relevant_num / np.float(all_num)
        #Recall.append(r)

    return np.mean(np.array(APx)), APx #np.mean(np.array(Recall)), APx


def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    video_path = []
    for i, (input, target, video_subpath) in enumerate(data_loader):
        input = input.float().cuda()
        target = target.cuda()
        y = model(input)
        for subpath in video_subpath:
            video_path.append(subpath)
        if is_start:
            all_output = y.data.cpu().float()
            all_label = target.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, y.data.cpu().float()), 0)
            all_label = torch.cat((all_label, target.float()), 0)

        if i%100==0:
            print('Generating, batch: %d'%i)

    return all_output.cpu().numpy(), all_label.cpu().numpy(), video_path


if __name__=='__main__':
    parser = argparse.ArgumentParser('Hash Test')
    parser.add_argument('--dataset', default='UCF101', choices=['UCF101', 'HMDB51'],
                        help="path to dataset")
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="define gpu id")
    # hash
    parser.add_argument('--hash_bit', type=int, default=64,
                        help="define the length of hash bits.")
    parser.add_argument('--batch_size', type=int, default=20)
    # initialization with priority (the next step will overwrite the previous step)
    # - step 1: random initialize
    # - step 2: load the 2D pretrained model if `pretrained_2d' is True
    # - step 3: load the 3D pretrained model if `pretrained_3d' is defined
    # - step 4: resume if `resume_epoch' >= 0
    parser.add_argument('--pretrained_2d', type=bool, default=True,
                        help="load default 2D pretrained model.")
    parser.add_argument('--pretrained_3d', type=str,
                        default='./exps/models/PyTorch-MFNet-master_ep-0013.pth',
                        help="load default 3D pretrained model.")
    parser.add_argument('--resume-epoch', type=int, default=-1,
                        help="resume train")
    parser.add_argument('--network', type=str, default='MFNet_3D',
                        choices=['MFNet_3D'],
                        help="chose the base network")

    # distributed training
    parser.add_argument('--backend', default='nccl', type=str, choices=['gloo', 'nccl'],
                        help='Name of the backend to use')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://192.168.0.11:23456', type=str,
                        help='url used to set up distributed training')
    # calculate MAP
    parser.add_argument('--R', default=100, type=int)
    parser.add_argument('--T', default=0, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    if args.dataset=='UCF101':
        args.R = 100
        Hash_center = torch.load('dataset/UCF101/raw/64_ucf_101_class.pkl')
        #args.pretrained_3d = './exps/models/PyTorch-MFNet-master_ep-0019_UCF101_2.pth'
    if args.dataset=='HMDB51':
        args.R = 70
        Hash_center = torch.load('dataset/HMDB51/raw/32_hmdb_51_class.pkl')
        #args.pretrained_3d = './exps/models/PyTorch-MFNet-master_ep-0100_HMDB51_5.pth'
    dataset_cfg = dataset.get_config(name=args.dataset)
    dataset_name = args.dataset

    batch_size = args.batch_size
    clip_length = 16
    train_frame_interval=2
    val_frame_interval=2
    distributed = False
    resume_epoch = -1
    iter_seed = int(torch.initial_seed()/1000000000000) \
                + (torch.distributed.get_rank() * 10 if distributed else 100) \
                + max(0, resume_epoch) * 100

    args.distributed = args.world_size > 1  # False


    net, input_conf = get_symbol(name=args.network,
                                 pretrained=args.pretrained_2d if args.resume_epoch < 0 else None,
                                 print_net=True if args.distributed else False,
                                 hash_bit = args.hash_bit,
                                 **dataset_cfg)
    net.eval()
    net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.pretrained_3d)
    net.load_state_dict(checkpoint['state_dict'])

    train_iter, eval_iter = iterator_factory.creat(name=dataset_name,
                                                   batch_size=batch_size,
                                                   clip_length=clip_length,
                                                   train_interval=train_frame_interval,
                                                   val_interval=val_frame_interval)
    print(len(train_iter))
    print(len(eval_iter))
    # print(net)
    print('Generating hash for database.............')
    database_hash, database_labels, dataset_path = predict_hash_code(net, train_iter)
    print(database_hash.shape)
    print(database_labels.shape)
    file_dir = 'dataset/' + args.dataset
    np.save(file_dir + '/database_hash.npy', database_hash)
    np.save(file_dir + '/database_label.npy', database_labels)
    np.save(file_dir + '/database_path.npy', dataset_path)
    print('Generating hash for test................')
    test_hash, test_labels, test_path = predict_hash_code(net, eval_iter)
    np.save(file_dir + '/test_hash.npy', test_hash)
    np.save(file_dir + '/test_label.npy', test_labels)
    np.save(file_dir + '/test_path.npy', test_path)
    print(test_hash.shape)
    print(test_labels.shape)

    print('Calculate MAP.....')
    MAP, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args)
    print(len(APx))
    print('MAP: %.4f' % MAP)
    #print('Recall:%.4f' % R)






