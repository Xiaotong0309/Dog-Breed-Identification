import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
# import seaborn as sns
import csv
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


train = pickle.load( open( "all/pre_p/train.p", "rb" ) )
print(train.shape)

test = pickle.load( open( "all/pre_p/test.p", "rb" ) )
print(test.shape)
train_label = pickle.load( open( "all/pre_p/train_label.p", "rb" ) )
print(train_label)
test_label = pickle.load( open( "all/pre_p/test_label.p", "rb" ) )
print(test_label)

label_binrizer = LabelBinarizer()
train_labels = label_binrizer.fit_transform(train_label)
test_labels = label_binrizer.fit_transform(test_label)

training_epochs = 20
batch_size = 128
examples_to_show = 10
dropout=0.2
# classnum=label_binrizer.classes_.size
classnum=120


net = Sequential()
net.add(Conv2D(64, kernel_size=(3,3), activation = 'relu',padding='same', input_shape=(28, 28 ,3) ,name='conv1'))
net.add(MaxPooling2D(pool_size = (2, 2),padding='same',name='pool1'))

net.add(Conv2D(64, kernel_size = (3, 3),padding='same', activation = 'relu',name='conv2'))
net.add(MaxPooling2D(pool_size = (2, 2),padding='same',name='pool2'))

net.add(Conv2D(64, kernel_size = (3, 3),padding='same', activation = 'relu',name='conv3'))
net.add(MaxPooling2D(pool_size = (2, 2),padding='same',name='pool3'))

net.add(Flatten())
net.add(Dense(128, activation = 'relu'))
net.add(Dropout(dropout))
net.add(Dense(classnum, activation = 'softmax'))

net.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
print(train.shape)
history = net.fit(train.reshape(-1, 28, 28, 3),train_labels, validation_data = (test[0:100].reshape(-1, 28, 28, 3),train_labels[0:100]) , epochs=training_epochs,shuffle = True, batch_size=batch_size)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.savefig("kaggle_padding.png")
plt.show()
