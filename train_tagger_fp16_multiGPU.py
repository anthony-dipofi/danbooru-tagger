import os

import numpy as np
import cv2
import torch
from PIL import Image
import torchvision
#from torchvision.transforms import ToTensor, ToPILImage
import random
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import torch.optim as optim
import math
from functools import reduce
from torch.utils.tensorboard import SummaryWriter
import atexit

import pandas as pd
import json
#from six_k_tags import *

from batch_sampler import *
import tagger_net

from apex import amp

#import amp





#net = tagger_net.res_tag_3(base_model = 'resnext101_32x8d', softmax_mixture_n = 16)
#net = tagger_net.res_tag_2(base_model = 'resnet50', out_classes = 6000)
#net = tagger_net.res_tag_2(base_model = 'resnet152')
#net = tagger_net.res_tag_2(base_model = 'resnet18', res_out_features = 512, out_classes = 6000)
net=torch.load("experiments/run11_resnet50_9/run11_resnet50_9_39999.pth").module


save_name = "run11_resnet50_10"

net = net.cuda()
batch_size = 512
val_batch_size = 512
val_thres = 0.001
val_thres_nn = 8.0

max_steps = 40000
lr_decay_step = 40000
lr_decay_factor = 1

num_classes = 6000
val_classes = 6000

learn_rate = 0.000003
opt_params = [{'params': list(net.parameters()), 'lr': learn_rate, 'weight_decay':0.00003}]
optimizer = optim.Adam(opt_params)

net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

net = nn.DataParallel(net)


#lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=max_steps)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_decay_step, gamma = lr_decay_factor)


class_weights = torch.FloatTensor([1, 10]).cuda() #approx 20 tags per image of 6000 possible, therefore weights positives *300



def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * (logsoftmax(pred)), dim=1))

loss_func = cross_entropy#torch.nn.NLLLoss(weight =  class_weights)#KLDivLoss()

net = net.cuda()#.half()

#folder = "d1k/d1k/"

#val_folder = "d1k/d1k_validation/"
#val_batch_size = 32
val_step = 10
#.pth"

run_folder = "experiments/"+save_name+"/"

save_step = 1000

train_data = json.load(open("metadata/image2idx_6k_train.json", 'r'))
train_data_by_class_idx = json.load(open("metadata/idx2image_6k_train.json", 'r'))

#print("y")

sampler_params = {
    "batch_size": batch_size,
    "dataset_metadata": train_data,
    "dataset_class": train_data_by_class_idx,
    "num_classes": 6000,
    "num_procs": 6,
    "max_queue_size" : 8,
    "target_mode": "softmax"
}


val_data = json.load(open("metadata/image2idx_6k_val.json", 'r'))
val_data_by_class_idx = json.load(open("metadata/idx2image_6k_val.json", 'r'))

val_sampler_params = {
    "batch_size": val_batch_size,
    "dataset_metadata": val_data,
    "dataset_class": val_data_by_class_idx,
    "num_classes": 6000,
    "num_procs": 2,
    "max_queue_size" : 3,
    "target_mode": "softmax"
}

bs = batch_sampler( sampler_params)
#bs = batch_sampler( val_sampler_params)
val_bs = batch_sampler(val_sampler_params)


writer = SummaryWriter(log_dir = run_folder)

#print("y2")
def cleanup():
    bs.close()
    val_bs.close()

#print("y3")
atexit.register(cleanup)

#print("y4")

for i in range(max_steps):
    print(i)
    batch = bs.next()
    net.train()
    #print(i)

    x = torch.FloatTensor(batch['x']).float()
    y = torch.FloatTensor(batch['y']).cuda()
    
    out = net(x)
    
    yp = out['out']
    optimizer.zero_grad()

    #print("yp",yp.shape)
    #print("y",y.shape)

    #ypt = F.softmax(yp[0,:100].float())
    #yt = y[0,:100].float()


    #print("yp",yp,yp.shape)
    #yp = yp.permute([0,2,1]).reshape([-1,2])
    #print("yp",yp,yp.shape)
    #print("y",y,y.shape)
    #y = y[:,1,:].reshape([-1])
 
    tag_l = loss_func(yp, y)#, y.cuda())


    '''ypt = yp[:,:].float()
    yt = y[:,:].float()

    yt_pos = (yt>val_thres).int()
    ypt_pos = (ypt>val_thres).int()

    pct_correct_counter = (ypt_pos + 2 * yt_pos)

    true_negative = (pct_correct_counter == 0).sum().float() / (val_classes * val_batch_size)
    false_positive = (pct_correct_counter == 1).sum().float() / (val_classes * val_batch_size)
    false_negative = (pct_correct_counter == 2).sum().float() / (val_classes * val_batch_size)
    true_positive = (pct_correct_counter == 3).sum().float() / (val_classes * val_batch_size)

    positives_correct = true_positive / (true_positive + false_negative)

    negatives_correct = true_negative / (true_negative + false_positive)


    writer.add_scalar('Train/true_negative', float(true_negative), i)
    writer.add_scalar('Train/false_negative', float(false_negative), i)
    writer.add_scalar('Train/false_positive', float(false_positive), i)
    writer.add_scalar('Train/true_positive', float(true_positive), i)
    writer.add_scalar('Train/correct', float(true_positive) + float(true_negative), i)
    writer.add_scalar('Train/positives_correct', float(positives_correct), i)
    writer.add_scalar('Train/negatives_correct', float(negatives_correct), i)'''

    l = tag_l

    with amp.scale_loss(l, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
    lr_scheduler.step()


    #print(ypt)
    #print(yt)

    #pct_correct = ((ypt>0.001).float() + 2 *(yt>0.001).float())



    writer.add_scalar('Loss/train_tag', float(tag_l), i)
    #writer.add_scalar('Loss/train_class', float(class_l), i)
    writer.add_scalar('Loss/train', float(l), i)
    #writer.add_scalar('Loss/pct_correct', 1.0 - float(pct_correct), i)
    print(l, bs.batch_queue.qsize(),"\n")

    if i % save_step == 0:
        os.makedirs(run_folder, exist_ok=True)
        torch.save(net, run_folder + save_name+"_"+str(i)+".pth")


    if i % val_step == 0:
        net.eval()
        with torch.no_grad():
            batch = val_bs.next()


            x = torch.FloatTensor(batch['x']).float()
            y = torch.FloatTensor(batch['y']).cuda()
            
            out = net(x)

            yp = out['out']
            ypts = yp[:,:val_classes].float()
            ypt = F.softmax(yp[:,:val_classes], dim =1).float()
            yt = y[:,:val_classes].float()

            val_l = loss_func(yp, y)


            print("ypt", ypt)
            print("yt", yt)
            
            yt_pos = (yt>val_thres).int()
            ypt_pos = (ypt>val_thres).int()

            ypts_pos = (ypts>val_thres_nn).int()

            pct_correct_counter = (ypt_pos + 2 * yt_pos)
            pct_nn_counter = (ypts_pos + 2 * yt_pos)

            print("pct", pct_correct_counter[0,:50])

            true_negative = (pct_correct_counter == 0).sum().float() / (val_classes * val_batch_size)
            false_positive = (pct_correct_counter == 1).sum().float() / (val_classes * val_batch_size)
            false_negative = (pct_correct_counter == 2).sum().float() / (val_classes * val_batch_size)
            true_positive = (pct_correct_counter == 3).sum().float() / (val_classes * val_batch_size)


            true_negative_nn = (pct_nn_counter == 0).sum().float()
            false_positive_nn = (pct_nn_counter == 1).sum().float() 
            false_negative_nn = (pct_nn_counter == 2).sum().float() 
            true_positive_nn = (pct_nn_counter == 3).sum().float() 

            positives_correct = true_positive / (true_positive + false_negative)

            negatives_correct = true_negative / (true_negative + false_positive)

            recall = true_positive_nn / (true_positive_nn + false_negative_nn)
            precision = true_positive_nn / (true_positive_nn + false_positive_nn)


            F_score = 2 * (precision * recall)/(precision + recall)


            print('true_positive', float(true_positive))
            print('true_negative', float(true_negative))
            print('false_positive', float(false_positive))
            print('false_negative', float(false_negative))

            print('true_positive_nn', float(true_positive_nn))
            print('true_negative_nn', float(true_negative_nn))
            print('false_positive_nn', float(false_positive_nn))
            print('false_negative_nn', float(false_negative_nn))

            print('positives_correct', float(positives_correct))
            print('negatives_correct', float(negatives_correct))

            print('precision', float(precision))
            print('recall', float(recall))
            print('F_score', float(F_score))


            print('correct', float(true_positive) + float(true_negative))

            #val_loss = loss_func(out, batch['y'].cuda())
            writer.add_scalar('Val/true_negative', float(true_negative), i)
            writer.add_scalar('Val/false_negative', float(false_negative), i)
            writer.add_scalar('Val/false_positive', float(false_positive), i)
            writer.add_scalar('Val/true_positive', float(true_positive), i)
            writer.add_scalar('Val/correct', float(true_positive) + float(true_negative), i)
            writer.add_scalar('Val/precision', float(precision), i)
            writer.add_scalar('Val/recall', float(recall), i)
            writer.add_scalar('Val/F_score', float(F_score), i)
            writer.add_scalar('Val/positives_correct', float(positives_correct), i)
            writer.add_scalar('Val/negatives_correct', float(negatives_correct), i)
            writer.add_scalar('Val/loss', float(val_l), i)

torch.save(net, run_folder + save_name+"_"+str(i)+".pth")

    
    