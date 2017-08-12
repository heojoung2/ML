# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import h5py

def Classificaiton_list():
    list = []
    names = os.listdir(train_path)

    for name in names:
        list.append(name)
    return list

def Classification_dir():
    dir={}

    classification_num = 0
    for category in classification_list:
        dir[category] = classification_num
        classification_num += 1

    return dir

image_size = 224
train_path = "C:/users/heojo/Desktop/Colorization/Image_data/"
classification_list= Classificaiton_list()
classification_dir = Classification_dir()

X = []
Y_colorization = []
Y_classification = []

for category in classification_list[:1]:
    names = os.listdir(train_path+category)
    for name in names:
        image = cv2.imread(train_path + category +'/' + name)
        image = cv2.resize(image,(image_size,image_size) ,interpolation = cv2.INTER_AREA)

        input_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        output_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        output_image_ab = output_image[:,:,1:]

        X.append(input_image)
        Y_colorization.append(output_image_ab)
        Y_classification.append(classification_dir[category])

'''
with h5py.File('train_data.hf','w') as hf:
    hf.create_dataset("X",data=X)
    hf.create_dataset("Y_colorization", data=Y_colorization)
    hf.create_dataset("Y_classification", data=Y_classification)
'''