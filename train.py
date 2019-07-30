from options import parser
from network import Model, AlexNetFc
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import numpy as np
#from pairwise_loss import pairwise_loss
from my_pairwise_loss import pairwise_loss
import os
from test import mean_average_precision, predict_hash_code

def main(args):
    global data_imbalance
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    if args.data_name == 'imagenet':
        train_list = 'data/imagenet/train.txt'
        test_list = 'data/imagenet/test.txt'
        #true_hash = 'data/imagenet/imagenet_100_class.pkl'
        true_hash = 'data/imagenet/hash_centers/' + str(args.hash_bit) + '_imagenet_100_class.pkl'
        #true_hash = 'data/imagenet/32_imagenet_100_class.pkl'
        data_imbalance = 100
        two_loss_epoch = -1
        total_epoch = 90

    elif args.data_name == 'coco':
        train_list = 'data/coco/train.txt'
        test_list = 'data/coco/test.txt'
        #true_hash = 'data/coco/coco_ha80_class.pkl'
        true_hash = 'data/coco/hash_centers/' + str(args.hash_bit) + '_coco_80_class.pkl'
        data_imbalance = 1
        two_loss_epoch = -1
        total_epoch = 90

    elif args.data_name == 'nus_wide':
        train_list = 'data/nus_wide/train.txt'
        #true_hash = 'data/nus_wide/nus_wide_ha21_class.pkl'
        true_hash = 'data/nus_wide/hash_centers/' + str(args.hash_bit) + '_nus_wide_21_class.pkl'
        data_imbalance = 5
        two_loss_epoch = -1
        total_epoch = 90

    database_list = 'data/' + args.data_name + '/database.txt'
    test_list = 'data/' + args.data_name + '/test.txt'
    database = ImageList(open(database_list).readlines(),
                         transform=prep.image_test(resize_size=255, crop_size=224))
    database_loader = torch.utils.data.DataLoader(database, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers)

    test_dataset = ImageList(open(test_list).readlines(), transform=prep.image_test(resize_size=255, crop_size=224))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    print(true_hash)
    Hash_center = torch.load(true_hash)
    global random_center
    random_center = torch.randint_like(Hash_center[0], 2)
    #Hash_center[Hash_center < 0] = 0  # Hash centers are {0,1}, no this line Hash center are {-1,1}

    train_data = ImageList(open(train_list).readlines(),
                           transform=prep.image_train(resize_size=255, crop_size=224))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                                               shuffle=True, num_workers=10)
    if args.model_type == 'resnet50' or args.model_type == 'resnet152':
        model = Model(args).cuda()
    elif args.model_type =='Alexnet':
        model = AlexNetFc(args).cuda()

    criterion = nn.BCELoss().cuda()
    #criterion = nn.MSELoss().cuda()
    params_list = [{'params': model.feature_layers.parameters(), 'lr': args.multi_lr*args.lr}, # 0.05*(args.lr)
                   {'params': model.hash_layer.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr = args.lr, betas=(0.9, 0.999))

    #if len(args.gpu_ids)>1:
        #model =  torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = torch.nn.DataParallel(model).cuda()

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>Start Train>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    best_MAP = 0
    for epoch in range(total_epoch):
        train(model, args, train_loader, criterion, Hash_center, optimizer, epoch, two_loss_epoch)

        if epoch%5 == 0:
            print('Testing, epoch: %d'%epoch)
            MAP = test_MAP(model, database_loader, test_loader, args)
            if MAP > best_MAP:
                best_MAP = MAP
                file_dir = args.data_name
                dir_name = 'data/' + file_dir + '/' + 'models/' + str(args.hash_bit) + 'bit_' + str(epoch) + 'e_' + str("{:.4g}".format(MAP)) + '_' + args.model_type + '.pkl'
                torch.save(model, dir_name)
                print('save model in: %s'%dir_name)
            print('MAP:%.3f'%MAP)

def train(model, args, train_loader, criterion, Hash_center, optimizer, epoch, two_loss_epoch):
    lr = adjust_learning_rate(optimizer, epoch)
    model.train()
    start_time = time.time()
    iter_num = 0
    total_loss = []
    for i, (input, label) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.data_name == 'imagenet':
            hash_label = (label == 1).nonzero()[:, 1]
            hash_center = Hash_center[hash_label]
        elif args.data_name == 'nus_wide' or args.data_name == 'coco':
            hash_center = Hash_center_multilables(label, Hash_center)
            #hash_label = (torch.cumsum(torch.cumsum(label, dim=1), dim=1) == 1).nonzero()[:, 1] # obtain the index of first 1 element in every row
            #hash_center = Hash_center[hash_label]

        hash_center = Variable(hash_center).cuda()

        input = Variable(input).cuda()
        y = model(input)

        #y = y[torch.mean(label.float(), dim=1)!=0]  # ignore some training image whose label is all zeros, this is for nus_wide
        #hash_center = hash_center[torch.mean(label.float(), dim=1)!=0]

        center_loss = criterion(0.5 * (y + 1), 0.5 * (hash_center + 1))
        Q_loss = torch.mean((torch.abs(y)-1.0)**2)

        if epoch <= two_loss_epoch:
            loss = args.lambda0*center_loss + args.lambda2 * Q_loss
        else:
            if len(label) < args.batch_size:  # if the last batch is not a complete batch, just set similarity_loss=0
                similarity_loss = 0
                # loss = center_loss #+ loss_mean
            else:
                output1 = y.narrow(0, 0, int(0.5 * len(y)))
                output2 = y.narrow(0, int(0.5 * len(y)), int(0.5 * len(y)))
                label1 = label[0:int(0.5 * len(label))]  # shape: [1/2*batch_size, num_class]
                label2 = label[int(0.5 * len(label)):int(len(label))]  # shape: [1/2*batch_size, num_class]
                label1 = torch.autograd.Variable(label1).cuda()
                label2 = torch.autograd.Variable(label2).cuda()
                similarity_loss = pairwise_loss(output1, output2, label1, label2,
                                                sigmoid_param=10. / args.hash_bit,
                                                #l_threshold=15,  # "l_threshold":15.0,
                                                data_imbalance=data_imbalance)  # for imagenet, is 100
            loss = args.lambda0*center_loss + args.lambda1*similarity_loss + args.lambda2*Q_loss

        loss.backward()
        optimizer.step()
        iter_num += 1
        total_loss.append(loss.data.cpu().numpy())

        if i%100==0:
            end_time1 = time.time()
            print('epoch: %d, lr: %.5f iter_num: %d, time: %.3f, loss: %.3f' % (epoch, lr, iter_num,(end_time1-start_time), loss))


    end_epoch_time = time.time()
    epoch_loss = np.mean(total_loss)
    print('Epoch: %d, time: %.3f, epoch loss: %.3f' % (epoch, end_epoch_time-start_time, epoch_loss))
    #if epoch_loss <= 0.2:
        #file_dir = args.data_name
        #dir_name = 'data/' + file_dir + '/' + str(epoch_loss) + '_' + str(args.hash_bit) + '_' + args.model_type + '.pkl'
        #torch.save(model, dir_name)
    #print(y[0])
    #print(label[0])

def test_MAP(model, database_loader, test_loader, args):
    print('Waiting for generate the hash code from database')
    database_hash, database_labels = predict_hash_code(model, database_loader)
    print(database_hash.shape)
    print(database_labels.shape)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels = predict_hash_code(model, test_loader)
    print(test_hash.shape)
    print(test_labels.shape)
    print('Calculate MAP.....')
    MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args)

    return MAP

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.7 ** (epoch // 10))
    #for param_group in optimizer.param_groups:
        #param_group['lr'] = lr
    optimizer.param_groups[0]['lr'] = args.multi_lr*lr
    optimizer.param_groups[1]['lr'] = lr

    return lr

def Hash_center_multilables(labels, Hash_center): # label.shape: [batch_size, num_class], Hash_center.shape: [num_class, hash_bits]
    is_start = True
    for label in labels:
        one_labels = (label == 1).nonzero()  # find the position of 1 in label
        #if len(one_labels) == 0:    # In nus_wide dataset, some image's labels  are all zero, we ignore these images
            #Center_mean = torch.zeros((1, Hash_center.size(1))) # let it's hash center be zero
        #else:
        one_labels = one_labels.squeeze(1)
        Center_mean = torch.mean(Hash_center[one_labels], dim=0)
        Center_mean[Center_mean<0] = -1
        Center_mean[Center_mean>0] = 1
        #random_center = torch.randint_like(Hash_center[0], 2) # the random binary vector {0, 1}, has the same shape with label
        random_center[random_center==0] = -1   # the random binary vector become {-1, 1}
        Center_mean[Center_mean == 0] = random_center[Center_mean == 0]  # shape: [hash_bit]
        Center_mean = Center_mean.view(1, -1) # shape:[1,hash_bit]

        if is_start:  # the first time
            hash_center = Center_mean
            is_start = False
        else:
            hash_center = torch.cat((hash_center, Center_mean), 0)
            #hash_center = torch.stack((hash_center, Center_mean), dim=0)

    return hash_center


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

