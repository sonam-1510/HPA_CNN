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

train_ids, test_ids = train_test_split(train_labels, test_size=0.1, random_state=48) 
train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=48)


labels = [item.split() for item in train_labels['Target']]
mlb = MultiLabelBinarizer()
mlb.fit(labels)
classes = mlb.classes_
y_val = mlb.transform([item.split() for item in val_ids['Target']])
y_test = mlb.transform([item.split() for item in test_ids['Target']])

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
  
from imgaug import augmenters as iaa
def augment(image):
    augment_img = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(rotate=30),
            iaa.Affine(rotate=45),
            iaa.Affine(rotate=60),
            iaa.Scale(0.2),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug  
    
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
            
def train_generator(BATCH_SIZE, augment = True):
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
     
 def test_generator(BATCH_SIZE):
    image_folder = train_path
    while True:
        
        test_imgs = []
        test_labels = []
        
        for f in test_ids.values:
            img = get_rgb_img(image_folder,f[0])
            test_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            test_labels.append(f[1])
            if len(test_imgs) == BATCH_SIZE:
                imgs = np.stack(test_imgs, axis=0)
                labels = mlb.transform([item.split() for item in test_labels])
                if len(imgs.shape[ 1: ]) == 2:
                    imgs = np.expand_dims(imgs, axis=3)
                yield (imgs, labels)
                test_imgs =[]
                test_labels =[]
        if len(test_imgs) > 0:
            imgs = np.stack(test_imgs, axis=0)
            labels = mlb.transform([item.split() for item in test_labels])
            if len(imgs.shape[ 1: ]) == 2:
                imgs = np.expand_dims(imgs, axis=3)
            yield (imgs, labels)

DEPTH = 3
BATCH_SIZE = 32
IMG_SIZE = 512
SAMPLE_SHAPE = (IMG_SIZE, IMG_SIZE, DEPTH)

lr = 1e-3
adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator(BATCH_SIZE, augment = True),
                              steps_per_epoch = len(train_ids)/BATCH_SIZE,
                              validation_data=val_generator(BATCH_SIZE),
                              validation_steps=len(val_ids)/BATCH_SIZE,
                              epochs=30)
lw = 3
plt.figure(figsize=(10,6))
plt.plot(history.history['acc'], label = 'Training', marker = '*', linewidth = lw)
plt.plot(history.history['val_acc'], label = 'Validation', marker = 'o', linewidth = lw)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(fontsize = 'x-large')
plt.show()

lw = 3
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label = 'Training', marker = '*', linewidth = lw)
plt.plot(history.history['val_loss'], label = 'Validation', marker = 'o', linewidth = lw)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(fontsize = 'x-large')
plt.show()

predictions = model.predict_generator(test_generator(BATCH_SIZE),steps=len(test_ids)/BATCH_SIZE)

def binary_prf1(y_true, y_pred):
    
    num_pos = np.sum(y_true)
    pred_pos = np.sum(y_pred)
    tp = np.sum(y_true * y_pred)
    if pred_pos > 0:
        precision = tp/pred_pos
    else:
        precision = 0
    if num_pos > 0:
        recall = tp/num_pos
    else:
        recall = 0
        print('no pos cases for this class')
    if precision >0 or recall > 0:
        f1 = 2*precision*recall/(precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def max_thresh(y_val, predictions, n=100):
    x = np.linspace(0,1,n+1)[1 : -1]
    f1_matrix = np.zeros((len(x), 28))

    for i in range(28):
        class_f1 = []
        for thresh in x:
            pred_class = (predictions > thresh).astype(int)
            class_f1.append((binary_prf1(y_val[ :, i], pred_class[ : , i])[2]))
        f1_matrix[ :, i] = np.array(class_f1)
    max_loc = np.argmax(f1_matrix, axis=0)
    max_thresh = [x[i] for i in max_loc]
    return max_thresh

max_t = max_thresh(y_test, predictions)
pred_classes = (predictions > max_t).astype(int)

pred_labels = mlb.inverse_transform(pred_classes)
pred_labels = [' '.join(item) for item in pred_labels]

from sklearn.metrics import classification_report
label_names = ['0','1','2','3','4','5','6','7','8','9','10', '11','12','13','14','15','16','17','18', '19', '20', '21', '22','23','24','25','26','27']
print(classification_report(y_test, pred_classes,target_names=label_names))

y_test_dataframe = pd.DataFrame(y_test)

from sklearn.metrics import roc_auc_score
print('AUC CKECK-UP per CLASS')
classes= y_test_dataframe.columns
for i, n in enumerate(classes):
  print(classes[i])
  print(i, roc_auc_score(y_test[:, i], pred_classes[:, i]))
  print('---------')

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (10, 10))
for (i, label) in enumerate(classes):
    fpr, tpr, thresholds = roc_curve(y_test[:,i].astype(int), predictions[:,i])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')

from sklearn.metrics import multilabel_confusion_matrix
confusion = multilabel_confusion_matrix(y_test, pred_labels)






   
