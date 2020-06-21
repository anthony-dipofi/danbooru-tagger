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


class res_tagger(nn.Module):
    def __init__(self,
                 out_classes = 6000,
                 base_model = 'resnet50',
                 res_out_features = 2048):
        
        super(res_tagger, self).__init__()
        
        self.out_classes = out_classes
        self.res_out_features = res_out_features
        
        n = torch.hub.load('pytorch/vision:v0.6.0', base_model, pretrained=True)
        self.res_net  = nn.Sequential(*list(n.children())[:-1])

        self.out_1 = nn.Linear(res_out_features, res_out_features)
        self.out_2 = nn.Linear(res_out_features, out_classes)


    def forward(self, ins):

        #load images from batch
        v = ins

        #store batch size
        batch_size = len(v)
        
        #project resnet outputs to model dimension
        v = F.leaky_relu(self.res_net(v))[:,:,0,0]
        
        #print("v",v.shape)
        
        v = F.leaky_relu(self.out_1(v))
        v = self.out_2(v)
        
        v = v.reshape([batch_size, self.out_classes])
        
        v = torch.clamp(v, -10, 10)

        out_v = v#F.log_softmax(v, dim=1)

        return {"out": out_v}

        
  
class res_tag_2(nn.Module):
    def __init__(self,
                 model_features = 1000,
                 out_classes = 6000,
                 input_size = 256,
                 base_model = 'resnet50',
                 res_out_features = 2048):
        
        super(res_tag_2, self).__init__()
        
        self.features = model_features
        self.out_classes = out_classes
        self.res_out_features = res_out_features
        
        n = torch.hub.load('pytorch/vision:v0.6.0', base_model, pretrained=True)
        self.res_net  = nn.Sequential(*list(n.children())[:-1])

        self.out_1 = nn.Linear(res_out_features, res_out_features)
        self.out_2 = nn.Linear(res_out_features, out_classes)


    def forward(self, ins):

        #load images from batch
        v = ins#['imgs']

        #initialize cuda for images
        v = v.cuda()
        
        #store batch size
        batch_size = len(v)
        
        #project resnet outputs to model dimension
        v = F.leaky_relu(self.res_net(v))[:,:,0,0]
        
        #print("v",v.shape)
        
        v = F.leaky_relu(self.out_1(v))
        v = self.out_2(v)
        
        v = v.reshape([batch_size, self.out_classes])
        
        v = torch.clamp(v, -10, 10)

        out_v = v#F.log_softmax(v, dim=1)

        return {"out": out_v}


        
class res_tag_3(nn.Module):
    def __init__(self,
                 model_features = 1000,
                 out_classes = 6000,
                 input_size = 256,
                 base_model = 'resnet50',
                 res_out_features = 2048,
                 softmax_mixture_n = 8):
        
        super(res_tag_3, self).__init__()
        
        self.features = model_features
        self.out_classes = out_classes
        self.res_out_features = res_out_features
        
        self.softmax_mixture_n = softmax_mixture_n

        n = torch.hub.load('pytorch/vision:v0.6.0', base_model, pretrained=True)
        self.res_net  = nn.Sequential(*list(n.children())[:-1])

        self.out_1 = nn.Linear(res_out_features, res_out_features)
        self.out_2 = nn.Linear(res_out_features, out_classes * softmax_mixture_n + softmax_mixture_n)


    def forward(self, ins):

        #load images from batch
        v = ins#['imgs']

        #initialize cuda for images
        v = v.cuda()
        
        #store batch size
        batch_size = len(v)
        
        #project resnet outputs to model dimension
        v = F.leaky_relu(self.res_net(v))[:,:,0,0]
        
        #print("v",v.shape)
        
        v = F.leaky_relu(self.out_1(v))
        v = self.out_2(v)

        #print("v",v.shape)

        head_weights = v[:,:self.softmax_mixture_n].unsqueeze(2)

        head_weights = F.softmax(head_weights, dim = 1)

        #print(" head_weights", head_weights.shape)

        v = v[:,self.softmax_mixture_n:].reshape([batch_size, self.softmax_mixture_n, self.out_classes]).contiguous()

        v = torch.clamp(v, -10, 10)

        v = F.softmax(v, dim = 2)

        v = head_weights * v

        v = v.sum(dim = 1)

        #print("v",v.shape)
        
        v = v.reshape([batch_size, self.out_classes])
        

        out_v = v#F.log_softmax(v, dim=1)

        return {"out": out_v}#, "im1_class": im1_classes_v, "im2_class": im2_classes_v}
       