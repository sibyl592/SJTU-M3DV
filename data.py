import csv
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import keras
from keras.losses import categorical_crossentropy

#########分出训练集的样本数为4/5 1/5为test集  留出法
x_path='./data/train'
x_file=os.listdir(x_path)
x_filef_train=x_file[0:508]
#x_filef_test=x_file[373:465]
x_test_path='./data/test'

def get_dataset():
    x_return_train=[]
    x_return_test=[]
    x_name=pd.read_csv("train_val.csv") ['name']
    for i in range(len(x_filef_train)):
        x_file_temp=os.path.join(x_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.9+x_voxel*0.1
        x_return_train.append(x_temp[34:66,34:66,34:66])
   # for i in range(len(x_filef_test)):
   #     x_file_temp=os.path.join(x_path,x_name[i+373]+'.npz')
   #     x_voxel=np.array(np.load(x_file_temp)['voxel'])
   #     x_mask=np.array(np.load(x_file_temp)['seg'])
   #     x_temp=x_voxel*x_mask
   #     x_return_test.append(x_temp[34:66,34:66,34:66])
    return  x_return_train

def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)[0:508]
    #x_test_label=keras.utils.to_categorical(x_label,2)[373:465]
    return x_train_label

def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("Sample.csv") ['Id']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.9+x_voxel*0.1
        x_return.append(x_temp[34:66,34:66,34:66])        
    return x_return
    
def mixup_data(x1, y1,alpha,n):
    x2=np.zeros(np.shape(x1))
    y2=np.zeros(np.shape(y1),'float')
    x3=np.zeros(n)
    y3=np.zeros(n,'float')
    l=len(x1)
    indexs = np.random.randint(0, l, n)
    indexs2 = np.random.randint(0, l, n)
    for i in range(n):
        x2[i] = x1[indexs2[i]]*alpha+(1-alpha)*x1[indexs[i]]
        y2[i] = y1[indexs2[i]]*alpha+(1-alpha)*y1[indexs[i]]
        
    x3 = x2[:n]
    y3 = y2[:n]
    return x3, y3

def spin(x1,y1,n):
    x2 = np.zeros(n)
    y2=np.zeros(n,'float')
    l = len(x1)
    indexs = np.random.randint(0, l, n)
    indexs2 = np.random.randint(0, l, n)
    for i in range(n):
        x2[i,:,:,:] = np.rot90(x1[indexs[i],:,:,:],1)
        y2[i,:,:,:] = np.rot90(y1[indexs[i],:,:,:],1)
        
    return x2,y2
    
def reverse_spin(x1,y1,n):
    x2 = np.zeros(n)
    y2=np.zeros(n,'float')
    l = len(x1)
    indexs = np.random.randint(0, l, n)
    for i in range(n):
        x2[i,:,:,:] = np.rot90(x1[indexs[i],:,:,:], -1)
        y2[i,:,:,:] = np.rot90(y1[indexs[i],:,:,:], -1)
        
    return x2,y2
    
def leftright(x1,y1,n):
    x2 = np.zeros(n)
    y2=np.zeros(n,'float')
    l = len(x1)
    indexs = np.random.randint(0, l, n)
    for i in range(n):
        x2[i,:,:,:] = np.fliplr(x1[indexs[i],:,:,:])
        y2[i,:,:,:] = np.fliplr(y1[indexs[i],:,:,:])
    
def updown(x1,y1,n):
    x2 = np.zeros(n)
    y2=np.zeros(n,'float')
    l = len(x1)
    indexs = np.random.randint(0, l, n)
    for i in range(n):
        x2[i,:,:,:] = np.flipud(x1[indexs[i],:,:,:])
        y2[i,:,:,:] = np.flipud(y1[indexs[i],:,:,:])
    return x2,y2
