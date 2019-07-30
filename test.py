import numpy as np
import torchvision
import torch
from options import parser
from data_list import ImageList
import pre_process as prep
from torch.autograd import Variable


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
    #data_dir = 'data/' + args.data_name
    #ids_10 = ids[:10, :]

    #np.save(data_dir + '/ids.npy', ids_10)
    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx

def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    for i, (input, label) in enumerate(data_loader):
        input = Variable(input).cuda()
        label = Variable(label).cuda()
        y = model(input)

        if is_start:
            all_output = y.data.cpu().float()
            all_label = label.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, y.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.float()), 0)

    return all_output.cpu().numpy(), all_label.cpu().numpy()


def test_MAP(model,database_loader,test_loader, args):
    print('Waiting for generate the hash code from database')
    database_hash, database_labels = predict_hash_code(model, database_loader)
    file_dir = 'data/'+ args.data_name
    da_ha_name = args.data_name + '_' + str(args.hash_bit) + '_database_hash.npy'
    da_la_name = args.data_name + '_' + str(args.hash_bit) + '_database_label.npy'
    np.save(file_dir + '/'+ da_ha_name, database_hash)
    np.save(file_dir + '/'+ da_la_name, database_labels)
    print(database_hash.shape)
    print(database_labels.shape)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels = predict_hash_code(model, test_loader)
    te_ha_name = args.data_name + '_' + str(args.hash_bit) + '_test_hash.npy'
    te_la_name = args.data_name + '_' + str(args.hash_bit) + '_test_label.npy'
    np.save(file_dir + '/'+ te_ha_name, test_hash)
    np.save(file_dir + '/'+ te_la_name, test_labels)
    print(test_hash.shape)
    print(test_labels.shape)
    print('Calculate MAP.....')
    MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args)

    return MAP, R, APx



if __name__ == '__main__':
    args = parser.parse_args()
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    database_list = 'data/' + args.data_name + '/database.txt'
    test_list = 'data/' + args.data_name + '/test.txt'
    model_name = args.model_name   # or just put your model name here
    model_dir = 'data/' + args.data_name + '/' + model_name
    model = torch.load(model_dir)

    """
    if args.data_name == 'imagenet':
        database_list = 'data/imagenet/database.txt'
        test_list = 'data/imagenet/test.txt'
        num_class = 100
        model = torch.load('data/imagenet/64_Resnet152_center.pkl')

    elif args.data_name == 'coco':
        database_list = 'data/coco/database.txt'
        data_name = 'coco'
        test_list = 'data/coco/test.txt'
        num_class = 80
        model = torch.load('data/coco/64_Resnet152_center.pkl')

    elif args.data_name == 'nus_wide':
        database_list = 'data/nus_wide/database.txt'
        data_name = 'nus_wide'
        test_list = 'data/nus_wide/test.txt'
        num_class = 21
        model = torch.load('data/nus_wide/64_Resnet152_center.pkl')
    """

    database = ImageList(open(database_list).readlines(),
                         transform=prep.image_test(resize_size=255, crop_size=224))
    database_loader = torch.utils.data.DataLoader(database, batch_size = args.batch_size, shuffle=False, num_workers=args.workers)


    test_dataset = ImageList(open(test_list).readlines(), transform=prep.image_test(resize_size=255, crop_size=224))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.workers)

    print('>>>>>>>>>>>>>>>>>>Testing>>>>>>>>>>>>>>>>>>')
    MAP, R, APx = test_MAP(model, database_loader, test_loader, args)

    np.save('data/Apx.npy', np.array(APx))
    print(len(APx))
    print('MAP: %.4f' % MAP)
    print('Recall:%.4f' % R)
