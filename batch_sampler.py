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

import pandas as pd
import json
#from six_k_tags import *




def preprocess_image(img):
    img =  img.convert("RGB")
    transforms = torchvision.transforms.Compose(
        [
            #torchvision.transforms.RandomAffine(15),
            #torchvision.transforms.RandomGrayscale(p=1.0),
            #torchvision.transforms.RandomResizedCrop(256, 
            #                                         scale=(0.4, 1), 
            #                                         ratio=(0.75, 1.3333), 
            #                                         interpolation=2),
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            
            #torchvision.transforms.ColorJitter(brightness=0.1, 
            #                                   contrast=0.1, 
            #                                   saturation=0.1, 
            #                                   hue = 0.1),
            
            #t#,
            #torchvision.transforms.ToTensor()

        ]
    )
    img = transforms(img)
    img = np.array(img)
    
    img = np.moveaxis(img, -1, 0) * (1.0/255)
    #print(img.shape)
    return img
    
def load_image(filename):
    #img_shape = (0,0)
    #while (len(img_shape) != 3):
    img = Image.open(filename)
    #    img_shape = img.shape
    img = preprocess_image(img)
    
    return img

def sample_image(sp):
    #select the class uniformly at random
    im_class = random.randint(0,sp['num_classes']-1)
    #print("im_class",im_class)
    if not str(im_class) in sp['dataset_class']:
        return sample_image(sp)

    im_class_files = sp['dataset_class'][str(im_class)]
    
    #print("im_class_files",im_class_files)
    if( len(im_class_files) == 0):
        return sample_image(sp)

    #select the image in the class uniformly at random
    img_index = random.randint(0,len(im_class_files)-1)


    img_filename = im_class_files[img_index]
    img_num = img_filename.split("/")[-1].split(".")[0]


    ex = sp['dataset_metadata'][img_num]
    #print("ex",ex)
    #nm = str(img_index)
    #img_filename = "data/"+"0"+str(nm[-3:])+ "/" + nm + '.jpg'


    x = load_image(img_filename)

    if sp['target_mode'] == 'softmax':
        y = np.zeros((sp['num_classes']))
        y[ex] = 1.0/(len(ex)) 
    else:
        y = np.zeros((2, sp['num_classes']))
        y[0] = 1
        y[0,ex] = 0 
        y[1,ex] = 1 
    
    return x, y


def sample_batch(sp):
    xs = []
    ys = []
    for i in range(sp['batch_size']):
        x, y = sample_image(sp)
        xs.append(x)
        ys.append(y)
        
    x = np.stack(xs, axis = 0)
    y = np.stack(ys, axis = 0)

    return {'x': x, 'y': y}

def sampler_proc(queue, sp):
    while True:
        try:
            batch = sample_batch(sp)
        except Exception:
            print("exception!")
            continue
        queue.put(batch)

class batch_sampler():
    def __init__(self, sp):
        self.num_procs = sp['num_procs']
        self.batch_size = sp['batch_size']
        
        self.batch_queue = multiprocessing.Queue(sp['max_queue_size'])
                
        self.procs = []
        for i in range(self.num_procs):
            print("starting proc",i)
            new_p = multiprocessing.Process(target=sampler_proc, args = (self.batch_queue, sp))
            new_p.start()
            self.procs.append(new_p)
            
    def next(self):
        return self.batch_queue.get()     
    
    def close(self):
        for p in self.procs:
            p.kill()
            