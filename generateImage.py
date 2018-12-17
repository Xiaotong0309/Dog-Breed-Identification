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
archive_train = ZipFile("all/train.zip", 'r')
archive_test = ZipFile("all/test.zip", 'r')
labels_raw = pd.read_csv("all/labels1.csv", header=0, sep=',', quotechar='"')
#print(labels_raw['breed'][0])


#This line shows the number of images in the train database
print(len(archive_train.namelist()[:])-1) #we must remove the 1st value
def search_label(name):
    for i in range(len(labels_raw['id'])):
        str_name = 'train/'
        str_name += labels_raw['id'][i]
        str_name += '.jpg'
        #print(str_name)
        if str_name == name:
            return labels_raw['breed'][i]
def process_new_image(image, w, h, index, label_name):

    if index != -1:
        #save to train
        name = 'all/'
        name += str(index)
        name += '.jpg'
        print(name)
        image = image.rotate(index*15)
        image.save(name)
        #save to .csv
        myCsvRow = str(index) + ',' + label_name + '\n'
        with open('all/labels.csv','a') as fd:
            fd.write(myCsvRow)
        #print(label)
    
    image = image.resize((w, h))
    #plt.imshow(image)
    #plt.show()
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0) #255 = max of the value of a pixel
    return image


#normalize data
def DataBase_creator(archivezip, nwigth, nheight, save_name):
    #We choose the archive (zip file) + the new wigth and height for all the image which will be reshaped

    # Start-time used for printing time-usage below.
    start_time = time.time()

    s = ((len(archivezip.namelist()[:]) - 1)*5, nwigth, nheight,3) #nwigth x nheight = number of features because images are nwigth x nheight pixels

    allImage = np.zeros(s)

    crop_p = 1/20
    for i in range(1, len(archivezip.namelist()[:])):
      filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
      pic_name = archivezip.namelist()[i]
      print(pic_name)
      image = PIL.Image.open(filename) # open colour image

      #left up
      image1 = image.crop([0,0,image.size[0] * (1 - crop_p), image.size[1] * (1 - crop_p)])
      #image = image.rotate(15)
      #image = image.resize((227,227))

      # right up
      image2 = image.crop([image.size[0] * crop_p, 0, image.size[0], image.size[1] * (1 - crop_p)])

      # left down
      image3 = image.crop([ 0, image.size[1] * crop_p, image.size[0] * (1 - crop_p), image.size[1]])

      # right down
      image4 = image.crop([ image.size[0] * crop_p, image.size[1] * crop_p, image.size[0], image.size[1]])

      label_name = search_label(pic_name)
      print(label_name)
      image = process_new_image(image, nwigth, nheight, -1, label_name)
      image1 = process_new_image(image1, nwigth, nheight, (i-1)*5, label_name)
      image2 = process_new_image(image2, nwigth, nheight, (i-1)*5+1, label_name)
      image3 = process_new_image(image3, nwigth, nheight, (i-1)*5+2, label_name)
      image4 = process_new_image(image4, nwigth, nheight, (i-1)*5+3, label_name)

      break


    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))




image_resize = 40

DataBase_creator(archivezip = archive_train, nwigth = image_resize, nheight = image_resize , save_name = "train")
#DataBase_creator(archivezip = archive_test, nwigth = image_resize, nheight = image_resize , save_name = "test")


#test = pickle.load( open( "test.p", "rb" ) )
#print(test.shape)

#let's check one image from the train data base
lum_img = train[0,:,:,:]
plt.imshow(lum_img)
plt.show()
