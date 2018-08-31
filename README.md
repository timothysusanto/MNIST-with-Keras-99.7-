# MNIST-with-Keras-99.7-
THE MNIST DATABASE  of handwritten digits Yann LeCun, Courant Institute, NYU
Email me at tecknomart@yahoo.com if you need the MNIST_dataset.csv for train and test. 


1. Data pre-processing

1.1. Load data

1.2. Check shape, data type

1.3. Extract xtrain, ytrain

1.4. Mean and std of classes

1.5. Check nuls and missing values

1.6. Normalization

1.7. Reshape

1.8. One hot encoding of label

1.9. Split training and validation sets

2. CNN

2.1. Define model architecture

2.2. Compile

2.3. Set other parameters

2.5. Fit model



# import libraries

import numpy as np # linear algebra, matrix multiplications
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

Data pre-processing

1.1 Load data
  train:
this is the data used to train the CNN.
the image data and their corresponding class is provided.
the CNN learns the weights to create the mapping from the image data to their corresponding class.

  test:
this is the data used to test the CNN.
the image data and their corresponding class is provided.


train = pd.read_csv("../input/mnist_train.csv")
test = pd.read_csv("../input/mnist_test.csv")


1.2. Data type

  train:
the train dataframe contains data from 60k images.
the data from each image is streched out in 1D with 28*28 = 784 pixels.
the first column is the label/class it belongs to, the digit it represents.

  test:
the test dataframe contains data from 10k images.
this data shall be fed to the CNN so that it's new data, that the CNN has never seen before.
same as in the train dataset, image data is streched out in 1D with 784 pixels.


1.2 Extract xtrain, ytrain
The CNN will be fed xtrain and it will learn the weights to map xtrain to ytrain


# array containing labels of each image
y_train = train["label"]
print("Shape of ytrain: ", ytrain.shape)

# dataframe containing all pixels (the label column is dropped)
X_train = train.drop("label", axis=1)

# the images are in square form, so dim*dim = 784
from math import sqrt
dim = int(sqrt(xtrain.shape[1]))

print("The images are {}x{} squares.".format(dim, dim))
print("Shape of xtrain: ", xtrain.shape)

or
import tensorflow as tf
X-train = tf.reshape(X_train, [-1, 784])

The images are 28x28 squares.
Shape of xtrain:  (60000, 784)
This is call flatten the input data, which is in most cases below 95% of accuracy.

Instead, we want to use 3D shape of data which gives us 98% of accuracy.
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)




1.3. Mean and std of the classes

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# plot how many images there are in each class
sns.countplot(ytrain)

print(ytrain.shape)
print(type(ytrain))

# array with each class and its number of images
vals_class=ytrain.value_counts()
print(vals_class)

# mean and std
cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class,ddof=1)

print("The mean amount of elements per class is", cls_mean)
print("The standard deviation in the element per class distribution is", cls_std)

# 68% - 95% - 99% rule, the 68% of the data should be cls_std away from the mean and so on

if cls_std > cls_mean * (0.6827 / 2):
    print("The standard deviation is high")
    
# if the data is skewed then we won't be able to use accurace as its results will be misleading and we may use F-beta score instead.

Summary
Shape of xtrain is: (60000, 784)
Shape of ytrain is: (60000, )
Shape of test is: (10000, 784)



1.4. Check nuls and missing values
   
X_train.isnull().any().describe()
count       784
unique        1
top       False
freq        784
dtype: object
There are no missing values

X_test.isnull().any().describe()
count       784
unique        1
top       False
freq        784
dtype: object
There are no missing values



1.6. Normalization

Pixels are represented in the range [0-255], but the NN converges faster with smaller values, in the range [0-1] so they are normalized to this range.

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0



1.7. Reshape

# reshape of image data to (nimg, img_rows, img_cols, 1)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

Previous X_train shape, pixels are in 1D vector: (42000, 784)
After reshape, pixels are a 28x28x1 3D matrix: (42000, 28, 28, 1)

Previous X_test shape, pixels are in 1D vector: (28000, 784)
After reshape, pixels are a 28x28x1 3D matrix: (28000, 28, 28, 1)
Note
In real world problems, the dimensions of images could diverge from this particular 28x28x3 set in two ways:

Images are usually much bigger
In this case all images are 28x28x1, but in another problem I'm working on, I have images of 3120x4160x3, so much bigger and in RGB. Usually images are resized to much smaller dimensions, in my case I'm resizing them to 64x64x3 but they can be made much smaller depending on the problem. In this MNIST dataset there is no such problem since the dimensions are already small.

Images don't usually have the same dimensions
Different dimension images are a problem since dense layers at the end of the CNN have a fixed number of neurons, which cannot be dynamically changed. This means that the layer expects fixed image dimensions, which means all images must be resized to the same dimensions before training. There is another option, namely, using a FCN (fully convoluted network) which consits solely of convolutional layers and a very big pooling in the end, so each image can be of any size, but this architecture isn't as popular as the CNN + FC (fully connected) layers 
There are various methods to make images have the same dimensions:
  -resize to a fixed dimension
  -add padding to some images and resize
  
  

1.8. One hot encoding of label

At this point in the notebook the labels vary in the range [0-9] which is intuitive, but in order to define the type of loss for the NN later, which in this case is categorical_crossentropy (reason is explained in section 2), the targets should be in categorical format (=one hot-vectors): ex : 2 -> [0,0,1,0,0,0,0,0,0,0]

ytrain before
0 1
1 0
2 1
3 4
4 0

where the first column is the index,

ytrain after
[[0. 1. 0. ... 0. 0. 0.]
[1. 0. 0. ... 0. 0. 0.]
[0. 1. 0. ... 0. 0. 0.]
...
[0. 0. 0. ... 1. 0. 0.]
[0. 0. 0. ... 0. 0. 0.]
[0. 0. 0. ... 0. 0. 1.]]


from keras.utils.np_utils import to_categorical

print(type(ytrain))
# number of classes, in this case 10
nclasses = ytrain.max() - ytrain.min() + 1

print("Shape of ytrain before: ", ytrain.shape) # (60000,)

ytrain = to_categorical(ytrain, num_classes = nclasses)

print("Shape of ytrain after: ", ytrain.shape) # (60000, 10), also numpy.ndarray type
print(type(ytrain))



1.9. Split training and validation sets

The available data is 60k images. If the NN is trained with these 60k images, it might overfit and respond poorly to new data. Overfitting means that the NN doesn't generalize for the digits, it just learns the differences in those 60k images. When faced with new digits slightly different, the performance decreases considerably. This is not a good outcome, since the goal of the NN is to learn from the training set digits so that it does well on the new digits.

In order to avoid submitting the predictions and risking a bad performance, and to determine whether the NN overfits, a small percentage of the train data is separated and named validation data. The ratio of the split can vary from 10% in small datasets to 1% in cases with 1M images.

The NN is then trained with the remaining of the training data, and in each step/epoch, the NN is tested against the validation data and we can see its performance. That way we can watch how the loss and accuracy metrics vary during training, and in the end determine where there is overfitting and take action (more on this later). For example, the results I had after the 20th epoch with a certain CNN architecture which turned out to overfit:

loss: 0.0066 - acc: 0.9980 - val_loss: 0.0291 - val_acc: 0.9940
In this example and without getting much into detail, the training loss is very low while the val_loss is 4 times higher, and the training accuracy is a little higher than the val_acc. The accuracy difference is not that much, partly because we are talking about 0.998 vs 0.994, which is exceptionally high, but the difference in loss suggests an overfitting problem.

Coming back to the general idea, the val_acc is the important metric. The NN might do very well with trained data but the goal is that the NN learns to generalize other than learning the training data "by heart". If the NN does well with val data, it's probable that it generalizes well to a certain extent and it will do well with the test data. (more on this in section 2 regarding CNNs).

random_state in train_test_split ensures that the data is pseudo-randomly divided.
If the images were ordered by class, activating this feature guarantees their pseudo-random split.
The seed means that every time this pseudo-randomization is applied, the distribution is the same.

stratify in train_test_split ensures that there is no overrepresentation of classes in the val set.
It is used to avoid some labels being overrepresented in the val set.

Note: only works with sklearn version > 0.17

from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 2
np.random.seed(seed)

# percentage of xtrain which will be xval
split_pct = 0.1

# Split the train and the validation set
xtrain, xval, ytrain, yval = train_test_split(xtrain,
                                              ytrain, 
                                              test_size=split_pct,
                                              random_state=seed,
                                              stratify=ytrain
                                             )

print(xtrain.shape, ytrain.shape, xval.shape, yval.shape)


Summary
The available data is now divided as follows:

Train data: images (xtrain) and labels (ytrain), 90% of the available data
Validation data: images (xval) and labels (yval), 10% of the available data




2. CNN

In this section the CNN is defined, including architecture, optimizers, metrics, learning rate reductions, data augmentation... Then it is compiled and fit to the training set.

from keras import backend as K

# for the architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D

# optimizer, data generator and learning rate reductor
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


2.1. Define model architecture

Below is an example CNN architecture:

More info on CNN architectures here: How to choose CNN Architecture MNIST

Architecture layers

You can read about the theory of CNNs on the Internet from people more knowledgeable than me and who surely explain it much better. So I will skip the theory explanation for the Conv2D, MaxPool2D, Flatten and Dense layers and I will focus on smaller details.

Conv2D

filters: usually on the first convolutional layers there are less filters, and more deeper down the CNN. Usually a power of 2 is set, and in this case 16 offered poorer performance and I didn't want to make a big CNN with 64 or 128 filters for digit classification.

kernel_size: this is the filter size, usually (3,3) or (5,5) is set. I advise setting one, building the architecture and changing it to see if it affects the performance though it usually doesn't.

padding: two options

valid padding: no padding, the image shrinks after convolution: n - f + 1
same padding: padding of 2, the image doesn't shrink after convolution: p = (f-1)/2 → (n+2) - f(=3) + 1 = n
activation: ReLU is represented mathematically by max(0,X) and offers good performance in CNNs (source: the Internet)

MaxPool2D: the goal is to reduce variance/overfitting and reduce computational complexity since it makes the image smaller. two pooling options

MaxPool2D: extracts the most important features like edges
AvgPool2D: extracts smooth features

My personal conclusion then is that for binarized images, with noticeable edge differences, MaxPool performs better.

Dropout: you can read the theory on the Internet, it's a useful tool to reduce overfitting. The net becomes less sensitive to the specific weights of neurons and is more capable of better generalization and less likely to overfit to the train data. The optimal dropout value in Conv layers is 0.2, and if you want to implement it in the dense layers, its optimal value is 0.5: 

model = Sequential()

model.add(Conv2D(32,(5,5), padding='same',
                 activation='relu', input_shape =(28,28,1)))
model.add(Conv2D(32,(5,5), padding='same',
                activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3), padding='same',
                activation='relu'))
model.add(Conv2D(64,(3,3), padding='same',
                activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()



2.2. Compile the model

Optimizer: it represents the gradient descent algorithm, whose goal is to minimize the cost function to approach the minimum point. Adam optimizer is one of the best-performing algorithms: Adam: A Method for Stochastic Optimization. The default learning rate for the Adam optimizer is 0.001. Another optimizer choice may be RMSprop or SGD.

Loss function: It is a measure of the overall loss in the network after assigning values to the parameters during the forward phase so it indicates how well the parameters were chosen during the forward propagation phase. This loss function requires the labels to be encoded as one-hot vectors which is why this step was taken back in 1.8.

Metrics: this refers to which metric the network should achieve, the most common one being 'accuracy' but there are other metrics to measure the performance other than accuracy, such as precision or recall or F1 score. The choice depends on the problem itself. Where high recall means low number of false negatives , High precision means low number of false positives and F1 score is a trade off between them: Precision-Recall. Depending on the problem, accuracy may not be the best metric. Suppose a binary classification problem where there are much more 0 values than 1, and therefore it's crucial that the predicted 1's are mostly correct. A network that just outputs 0 every time would get very high accuracy but the model still wouldn't perform well. Take the popular example:

A ML company has built a tool to identify terrorists among the population and they claim to have 99.99% accuracy. When inspecting their product, turns out they just output 0 in every case. Since there is only one terrorist for every 10000 people (this is made up, I actually have no idea what the probability is but all I know is that it's very low), the company has a very high precision, but there's no need of a ML tool for that. With the class imbalance being so high, accuracy is not a good metric anymore and other options should be considered.

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



2.3. Set other parameters

Learning rate annealer

This is a useful tool which reduces the learning rate when there is a plateau on a certain value, which you can specify. In this case the monitoring value is val_acc. When there is no change in val_acc in 3 epochs (patience), the learning rate is multiplied by 0.5 (factor). If the learning rate has the value of min_lr, it stops decreasing.


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                 patience=3, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.00001)
Data augmentation

Data augmentation is a technique used to artificially make the training set bigger. There are a number of options for this, the most common ones include rotating images, zooming in a small range and shifting images horizontally and vertically.

Beware that activating some features may be confusing for the network, imagine that when taking img1 and flipping it, it may be very similar to img2 which has a different label. With the digits 6 and 9 for example, if you take either and flip it vertically and horizontally, it becomes the other. So if you do that with the digit 9, flip it in both edges and tell the network that the digit is still a 9 when it actually is very similar to the images of the digit 6, the performance will drop considerably. So take into account the images and how activating the features may affect the labeling.


datagen = ImageDataGenerator(
          featurewise_center=False,            # set input mean to 0 over the dataset
          samplewise_center=False,             # set each sample mean to 0
          featurewise_std_normalization=False, # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,                 # apply ZCA whitening
          rotation_range=30,                   # randomly rotate images in the range (degrees, 0 to 180)
          zoom_range = 0.1,                    # Randomly zoom image 
          width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
          horizontal_flip=False,               # randomly flip images
          vertical_flip=False)                 # randomly flip images

datagen.fit(xtrain)



Epochs and batch_size

Epochs: based on my experiments, the loss and accuracy get into a plateau at around the 10th epoch, so I usually set it to 15.
Batch_size: I skip the theory which you can read it on the Internet. I recommend that you try changing it and seeing the change in the loss and accuracy, in my case a batch_size of 16 turned out to be disastrous and the best case occurred when I set it to 64.

epochs = 20
batch_size = 64


2.4 Fit the model
Since there is data augmentation, the fitting function changes from fit (when there is no data augmentation) to fit_generator. The first input argument is slightly different. Otherwise you can specify the verbosity, number of epochs, validation data if any, any callbacks you want to include... This is one of the most time consuming cells in the notebook, and its running time depends on the number of epochs specified, number of trainable parameters in the network and input dimensions. Changing the batch_size also conrtibutes to changes in time, the bigger the batch_size, the faster the epoch.


Note: remember to create and compile the model all over again whenever you change something, such as batch_size or epochs or anything related to the CNN. if you don't and just run the fit cell, it will continue training on the old network.

history = model.fit_generator(datagen.flow(xtrain,ytrain, batch_size=batch_size),
                              epochs=epochs, 
                              validation_data=(xval,yval),
                              verbose=1, 
                              steps_per_epoch=xtrain.shape[0] // batch_size, 
                              callbacks=[lr_reduction])




Loss and accuracy

After training the model, it's useful to plot the loss and accuracy in training and validation to see its progress and detect problems. In this particular case with this particular network, the training loss decreases, which means the network is learning, and there is no substantial difference between the training loss and validation loss wich indicates no overfitting. At this levels where the loss is so low and accuracy is so high there really is no bias or variance problem, but if you want to improve results further you could approach a bias problem, in other words, that the training loss is too high. To reduce this the recommended solutions are making a bigger network and training for a longer time. Feel free to tweak the network or epochs.

