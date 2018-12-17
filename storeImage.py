import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#image_reshape, label_filter, one-hot, train_test_split
# Image manipulation.
import PIL.Image
from IPython.display import display
#from resizeimage import resizeimage

#Panda
import pandas as pd

#Open a Zip File
from zipfile import ZipFile
from io import BytesIO

#We unzip the train and test zip file
archive_train = ZipFile("all/train1.zip", 'r')
archive_test = ZipFile("all/test1.zip", 'r')
labels_raw = pd.read_csv("all/labels1.csv", header=0, sep=',', quotechar='"')

train_label = []
test_label = []

#img_num = 20
#This line shows the number of images in the train database
print(len(archive_train.namelist()[:])-1) #img_num #we must remove the 1st value
def search_label(name):
    for i in range(len(labels_raw['id'])):
#         str_name = 'train/'
        str_name = labels_raw['id'][i]
        str_name += '.jpg'
        #print(str_name)
        if str_name == name:
            return labels_raw['breed'][i]
    return labels_raw['breed'][i-1]

#normalize data
def DataBase_creator(archivezip, nwigth, nheight, save_name):
    #We choose the archive (zip file) + the new wigth and height for all the image which will be reshaped

    # Start-time used for printing time-usage below.
    start_time = time.time()

    s = (len(archivezip.namelist()[:])-1, nwigth, nheight,3) #nwigth x nheight = number of features because images are nwigth x nheight pixels
    allImage = np.zeros(s)

    for i in range(1,len(archivezip.namelist()[:])):#img_num
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        print(i, archivezip.namelist()[i])
        image = PIL.Image.open(filename) # open colour image
        image = image.resize((nwigth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) #255 = max of the value of a pixel

        allImage[i-1]=image

        pic_name = archivezip.namelist()[i]
        label_name = search_label(pic_name)
        if save_name == "train":

            train_label.append(label_name)
        else:
            test_label.append(label_name)


    #we save the newly created data base
    pickle.dump(allImage, open( save_name + '.p', "wb" ) )
    save_name += '_label'
    if save_name == "train_label":
        pickle.dump(train_label, open( save_name + '.p', "wb" ) )
    else: pickle.dump(test_label, open( save_name + '.p', "wb" ) )


    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

image_resize = 40

DataBase_creator(archivezip = archive_train, nwigth = image_resize, nheight = image_resize , save_name = "train")
DataBase_creator(archivezip = archive_test, nwigth = image_resize, nheight = image_resize , save_name = "test")

train = pickle.load( open( "train.p", "rb" ) )
print(train.shape)

test = pickle.load( open( "test.p", "rb" ) )
print(test.shape)
train_label = pickle.load( open( "train_label.p", "rb" ) )
print(train_label)
test_label = pickle.load( open( "test_label.p", "rb" ) )
print(test_label)
#print(len(train_label), train_label)
#print(len(test_label), test_label)
#let's check one image from the train data base
