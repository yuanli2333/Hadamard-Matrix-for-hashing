import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1, data_imbalance=1):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    loss = torch.mean(exp_loss)

    return loss