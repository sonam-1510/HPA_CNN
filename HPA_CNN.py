import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, CSVLogger
from keras.models import Model, Sequential, load_model, model_from_json
from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import random
import tensorflow as tf

data_path = '../input'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path,'test')
labels_path = os.path.join(data_path, 'train.csv')

train_labels = pd.read_csv(labels_path,index_col=False)
train_ids, val_ids = train_test_split(train_labels, test_size=0.2, random_state=48)
labels = [item.split() for item in train_labels['Target']]

mlb = MultiLabelBinarizer()
mlb.fit(labels)
classes = mlb.classes_
y_val = mlb.transform([item.split() for item in val_ids['Target']])

def model(sample_shape):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=sample_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(28))
    model.add(Activation('sigmoid'))
    return model
  
  def get_rgb_img(image_folder,img_id):
    img = []
    img.append(plt.imread(os.path.join(image_folder,img_id+'_red.png')))
    img.append(plt.imread(os.path.join(image_folder, img_id+'_blue.png')))
    img.append(plt.imread(os.path.join(image_folder, img_id+'_green.png')))
    return np.stack(img, axis=2)
  
  def val_generator(BATCH_SIZE):
    image_folder = train_path
    while True:
        val_imgs = []
        val_labels = []
        for f in val_ids.values:
            img = get_rgb_img(image_folder,f[0])
            val_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            val_labels.append(f[1])
            if len(val_imgs) == BATCH_SIZE:
                imgs = np.stack(val_imgs, axis=0)
                labels = mlb.transform([item.split() for item in val_labels])
                if len(imgs.shape[ 1: ]) == 2:
                    imgs = np.expand_dims(imgs, axis=3)
                yield (imgs, labels)
                val_imgs =[]
                val_labels =[]
        if len(val_imgs) > 0:
            imgs = np.stack(val_imgs, axis=0)
            labels = mlb.transform([item.split() for item in val_labels])
            if len(imgs.shape[ 1: ]) == 2:
                imgs = np.expand_dims(imgs, axis=3)
            yield (imgs, labels)
            
def train_generator(BATCH_SIZE):
    image_folder = train_path
    while True:
             
        train_imgs = []
        train_labels = []
        
        for f in train_ids.values:
            img = get_rgb_img(image_folder,f[0])
            train_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            train_labels.append(f[1])
            if len(train_imgs) == BATCH_SIZE:
                imgs = np.stack(train_imgs, axis=0)
                labels = mlb.transform([item.split() for item in train_labels])
                if len(imgs.shape[ 1: ]) == 2:
                    imgs = np.expand_dims(imgs, axis=3)
                yield (imgs, labels)
                train_imgs = []
                train_labels = []
        if len(train_imgs) > 0:
            imgs = np.stack(train_imgs, axis=0)
            labels = mlb.transform([item.split() for item in train_labels])
            if len(imgs.shape[ 1: ]) == 2:
                imgs = np.expand_dims(imgs, axis=3)
            
            yield (imgs, labels)
DEPTH = 3
BATCH_SIZE = 32
IMG_SIZE = 512
SAMPLE_SHAPE = (IMG_SIZE, IMG_SIZE, DEPTH)

lr = 1e-3
adam = Adam(lr=lr)
model.compile(optimizer=adam, 
                  loss='binary_crossentropy',
                metrics=['acc'])

history = model.fit_generator(train_generator(BATCH_SIZE),
                                  steps_per_epoch = len(train_ids)/BATCH_SIZE,
                                  validation_data=val_generator(BATCH_SIZE),
                                  validation_steps=len(val_ids)/BATCH_SIZE,
                                  epochs=30)

predictions = model.predict_generator(val_generator(BATCH_SIZE),steps=len(val_ids)/BATCH_SIZE)

   
