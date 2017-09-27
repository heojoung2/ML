# -*- coding: utf-8 -*-
import os
import csv

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

train_path = "D:/images256/"
classification_list= Classificaiton_list()
classification_dir = Classification_dir()

csv_file = open("train_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

X = []
Y_colorization = []
Y_classification = []

for category in classification_list:
    names = os.listdir(train_path+category)
    for name in names:
        path = train_path + category +'/' + name
        csv_writer.writerow([path,classification_dir[category]])

csv_file.close()
