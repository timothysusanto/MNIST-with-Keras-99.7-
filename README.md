# MNIST-with-Keras-99.7-
THE MNIST DATABASE  of handwritten digits Yann LeCun, Courant Institute, NYU


import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization

sns.set(style='white', context='notebook', palette='deep')

X_train = pd.read_csv('./mnist_train.csv')
X_test = pd.read_csv('./mnist_test.csv')

y_train = X_train['label']
X_train = X_train.drop(labels=['label'], axis=1)
g = sns.countplot(y_train)

y_test = X_test['label']
X_test = X_test.drop(labels=['label'], axis=1)
g = sns.countplot(y_test)


y_train.value_counts()
Out[20]:
1    6742
7    6265
3    6131
2    5958
9    5949
0    5923
6    5918
8    5851
4    5842
5    5421
Name: label, dtype: int64


y_test.value_counts()
Out[21]:
1    1135
2    1032
7    1028
3    1010
9    1009
4     982
0     980
8     974
6     958
5     892
Name: label, dtype: int64


X_train.isnull().any().describe()
Out[22]:
count       784
unique        1
top       False
freq        784
dtype: object


X_test.isnull().any().describe()
Out[23]:
count       784
unique        1
top       False
freq        784
dtype: object


#Normalization
X_train = X_train / 255
X_test = X_test / 255

#Reshape
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


#Encode labels to one hot vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_train, num_classes=10)


#split training and validation set for the fitting
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                 y_train,
                                                 test_size=0.1,
                                                 random_state=random_seed)


#  CNN model
model = Sequential()

model.add(Conv2D(32,kernel_size=3,input_shape=(28,28,1),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 32)        9248      
_________________________________________________________________
batch_normalization_2 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 32)        25632     
_________________________________________________________________
batch_normalization_3 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        18496     
_________________________________________________________________
batch_normalization_4 (Batch (None, 14, 14, 64)        256       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 64)        36928     
_________________________________________________________________
batch_normalization_5 (Batch (None, 14, 14, 64)        256       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 7, 7, 64)          102464    
_________________________________________________________________
batch_normalization_6 (Batch (None, 7, 7, 64)          256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               401536    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 597,066
Trainable params: 596,490
Non-trainable params: 576
_________________________________________________________________


epochs = 20
batch_size = 128


#data augmentation
datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2)
datagen.fit(X_train)


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)
                                           
                                           

history = model.fit_generator(datagen.flow(x = X_train, y = y_train, batch_size = batch_size),
                   epochs = epochs, validation_data = (X_val, y_val),
                   verbose = 2, 
                   steps_per_epoch = X_train.shape[0] // batch_size,
                   callbacks=[learning_rate_reduction])
                   
                   
Epoch 1/20
 - 332s - loss: 0.7521 - acc: 0.7558 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 2/20
 - 330s - loss: 0.2193 - acc: 0.9358 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 3/20
 - 332s - loss: 0.1569 - acc: 0.9551 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 4/20
 - 349s - loss: 0.1261 - acc: 0.9639 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00004: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.


Epoch 5/20
 - 365s - loss: 0.0899 - acc: 0.9743 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 6/20
 - 384s - loss: 0.0834 - acc: 0.9761 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 7/20
 - 404s - loss: 0.0786 - acc: 0.9777 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00007: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.


Epoch 8/20
 - 429s - loss: 0.0633 - acc: 0.9818 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 9/20
 - 434s - loss: 0.0591 - acc: 0.9828 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 10/20
 - 415s - loss: 0.0580 - acc: 0.9836 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.


Epoch 11/20
 - 412s - loss: 0.0533 - acc: 0.9848 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 12/20
 - 399s - loss: 0.0487 - acc: 0.9858 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 13/20
 - 397s - loss: 0.0507 - acc: 0.9860 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00013: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.


Epoch 14/20
 - 397s - loss: 0.0464 - acc: 0.9869 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 15/20
 - 392s - loss: 0.0442 - acc: 0.9872 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 16/20
 - 389s - loss: 0.0441 - acc: 0.9874 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00016: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.


Epoch 17/20
 - 428s - loss: 0.0408 - acc: 0.9886 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 18/20
 - 430s - loss: 0.0396 - acc: 0.9890 - val_loss: 14.5332 - val_acc: 0.0983
Epoch 19/20
 - 437s - loss: 0.0438 - acc: 0.9877 - val_loss: 14.5332 - val_acc: 0.0983

Epoch 00019: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.


Epoch 20/20
 - 399s - loss: 0.0397 - acc: 0.9889 - val_loss: 14.5332 - val_acc: 0.0983
