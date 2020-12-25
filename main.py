import os
import argparse
import tqdm
import os
import argparse
import numpy as np
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import time 
from utils import weights_init, print_args
from model import *

import scipy.io
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='./CWRU_dataset/')
parser.add_argument("--source", default='DE')
parser.add_argument("--target", default='FE')
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--shuffle", default=True, type=bool)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--class_num", default=3)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--weight_entropy", default=0.1, type=float)
parser.add_argument("--dropout_p", default=0.1, type=float)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--result", default='record')
parser.add_argument("--save", default=False, type=bool)
parser.add_argument("--lambda_val", default=1.0, type=float)
parser.add_argument("--entropy_thres", default=0.00000001, type=float)
parser.add_argument('--thres_rec', type=float, default=0.0001, help='coefficient for reconstruction loss')
parser.add_argument("--optimizer", default='Adam', type=str)
parser.add_argument('--GPU', type=bool, default=True,
                    help='enable train on GPU or not, default is False')



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    # resize for CWRU dataset
    source = source.reshape(int(source.size(0)), int(source.size(1))* int(source.size(2)))
    target = target.reshape(int(target.size(0)), int(target.size(1))* int(target.size(2)))
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(int(kernel_num))]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMDLoss(source, target):
    kernel_num = 2.0
    kernel_mul = 5
    fix_sigma = None
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def minmax_norm(data):
    min_v = np.min(data)
    range_v = np.max(data) - min_v
    data = (data - min_v) / range_v
    return data

# classification loss
def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

# compute entropy loss
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(args.entropy_thres)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    
    return args.weight_entropy * (entropy / float(p_softmax.size(0)))

# compute entropy
def HLoss(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum()
    return b

def load_data(domain):
    input_domain = np.load(args.data_root+'CWRU_'+domain+'.npy', allow_pickle=True)
    input_domain = input_domain.item()
    input_N = input_domain['Normal']
    input_OR = input_domain['OR']
    input_IR = input_domain['IR']
    # print (np.shape(input_IR), np.shape(input_OR), np.shape(input_N))

    input_label_N = np.zeros([np.size(input_N,0),1])
    input_label_OR = np.ones([np.size(input_OR,0),1])
    input_label_IR = np.ones([np.size(input_IR,0),1])+1

    data = np.concatenate((input_N, input_OR, input_IR) , axis=0)
    print(np.shape(data))
    label = np.concatenate((input_label_N, input_label_OR, input_label_IR), axis=0)
    print(np.shape(label))
    # shuffle inputs
    nums = [x for x in range(np.size(data, axis = 0))]
    random.shuffle(nums)
    data = data[nums, :]
    label = label[nums, :]

    data = np.transpose(data, (0, 2, 1))
    label = np.squeeze(label)
    return data, label
if __name__ == "__main__":
    args = parser.parse_args()
    print_args(args)

    t = time.time()
    # load source data
    source_data, source_label = load_data(args.source)
    # load target data
    target_data, target_label = load_data(args.target)

    # fead data to dataloder
    source_data = Variable(torch.from_numpy(source_data).float(), requires_grad=False)
    source_label= Variable(torch.from_numpy(source_label).long(), requires_grad=False)
    target_data = Variable(torch.from_numpy(target_data).float(), requires_grad=False)
    target_label= Variable(torch.from_numpy(target_label).long(), requires_grad=False)
    source_dataset = TensorDataset(source_data, source_label)
    target_dataset = TensorDataset(target_data, target_label)
    source_loader = DataLoader(source_dataset,batch_size=args.batch_size)
    target_loader = DataLoader(target_dataset,batch_size=args.batch_size)
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)

    # initialize model
    netG = Generator(source='CWRU_'+args.source, target='CWRU_'+args.target)
    netF = Classifier(source='CWRU_'+args.source, target='CWRU_'+args.target)
    if args.GPU:
        netG.cuda()
        netF.cuda()
    netG.apply(weights_init)
    netF.apply(weights_init)



    print ('Training using Adam')
    opt_g = optim.Adam(netG.parameters(), lr=args.lr, weight_decay=0.0005)
    opt_f = optim.Adam(netF.parameters(), lr=args.lr, weight_decay=0.0005)

    max_correct = -1.0
    correct_array = []

    # start training
    for epoch in range(1, args.epoch+1):
        source_loader_iter = iter(source_loader)
        target_loader_iter = iter(target_loader)
        print(">>training " + args.task + " epoch : " + str(epoch))

        netG.train()
        netF.train()
        tic = time.time()
        for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
            try:
                s_imgs, s_labels = source_loader_iter.next()
            except:
                source_loader_iter = iter(source_loader)
                s_imgs, s_labels = source_loader_iter.next()

            if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
                continue

            if args.GPU:
                s_imgs = Variable(s_imgs.cuda())
                s_labels = Variable(s_labels.cuda())     
                t_imgs = Variable(t_imgs.cuda())

            opt_g.zero_grad()
            opt_f.zero_grad()

            # apply feature extractor to input images
            s_bottleneck = netG(s_imgs) 
            t_bottleneck = netG(t_imgs)

            # get classification results
            s_logit = netF(s_bottleneck)
            t_logit = netF(t_bottleneck)
            t_logit_entropy = HLoss(t_bottleneck)
            s_logit_entropy = HLoss(s_bottleneck)

            # get source domain classification error
            s_cls_loss = get_cls_loss(s_logit, s_labels)
            
            # compute entropy loss
            t_prob = F.softmax(t_logit)
            t_entropy_loss = get_entropy_loss(t_prob)

            # MMFD loss
            MMD = MMDLoss(s_bottleneck, t_bottleneck)
            
            # Full loss function
            loss = s_cls_loss + t_entropy_loss + args.lambda_val*MMD - args.thres_rec*(t_logit_entropy +s_logit_entropy)
            
            loss.backward()
            
            if (i+1) % 50 == 0:
                print ("cls_loss: %.4f, MMD:  %.4f, t_HLoss:  %.4f, s_HLoss:  %.4f" % (s_cls_loss.item(), args.lambda_val*MMD.item(), args.thres_rec*t_logit_entropy.item(), args.thres_rec*s_logit_entropy.item()))
            
            opt_g.step()
            opt_f.step()
        print('Training time:', time.time()-tic)

        # evaluate model
        tic = time.time()
        netG.eval()
        netF.eval()
        correct = 0
        t_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        for (t_imgs, t_labels) in t_loader:
            if args.GPU:
                t_imgs = Variable(t_imgs.cuda())
            t_bottleneck = netG(t_imgs)
            t_logit = netF(t_bottleneck)
            pred = F.softmax(t_logit)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            t_labels = t_labels.numpy()
            correct += np.equal(t_labels, pred).sum()
        t_imgs = []
        t_bottleneck = []
        t_logit = []
        pred = []
        t_labels = []

        # compute classification accuracy for target domain
        correct = correct * 1.0 / len(target_dataset)
        correct_array.append(correct)

        if correct >= max_correct:
            max_correct = correct
        print('Test time:', time.time()-tic)
        print ("Epoch {0} accuray: {1}; max acc: {2}".format(epoch, correct, max_correct))

    # save results
    print("max acc: ", max_correct)
    max_correct = float("{0:.3f}".format(max_correct))
    result = open(os.path.join(args.result, "FRAN_" + args.task + "_" + str(max_correct) +"_lr_"+str(args.lr)+'_lambda_' + str(args.lambda_val) + '_recons_' + str(args.thres_rec)+"_weight_entropy_"+str(args.weight_entropy)+".txt"), "a")
    for c in correct_array:
        result.write(str(c) + "\n")
    result.write("Max: "+ str(max_correct) + "\n")
    elapsed = time.time() - t
    print("elapsed: ", elapsed)
    result.write(str(elapsed) + "\n")
    result.close()