import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hotencoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# *load_Data*
# loading data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# display first five rows of train_data
train_data.head()
test_data.head()
# checking shape of train_data
train_data.shape
# checking shape of test_data
test_data.shape
# *Check for null and missing values*
# check the data
train_data.describe()
# check missing and null values
test_data.isnull().sum()
train_data.isnull().sum()
Y_train = train_data["label"]
# Drop 'label' columnX_train = train_data.drop(labels = ["label"],axis
= 1)
# free some space
del train_data
g = sns.countplot(Y_train)
Y_train.value_counts()
# There is no missing values in the train and test dataset. So we can
safely go ahead.
# *Normalization*
# We perform a grayscale normalization to reduce the effect of
illumination's differences.
#
# Moreover the CNN converg faster on [0..1] data than on [0..255].
# Normalize the data
X_train= X_train / 255.0
test_data= test_data / 255.0
# *Reshape*
# In[ ]:# Reshape image in 3 dimensions (height = 28px, width = 28px ,
channel = 1)
X_train = X_train.values.reshape((-1,28,28,1))
test_data = test_data.values.reshape((-1,28,28,1))
test_data.shape
# *label_encoding*
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# *Split training and valdiation set*
# Set the random seed
random_seed = 2
# Split the train and the validation set for the fittingX_train,
X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size =
0.1, random_state=random_seed)
# i choosed to split the train set in two parts : a small fraction
(10%) became the validation set which the model is evaluated and the
rest (90%) is used to train the model.
# Some examples
import matplotlib.pyplot as plt
h = plt.imshow(X_train[0][:,:,0])
k = plt.imshow(X_train[10][:,:,0])
# *Introduction to convnets*
from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Let’s display the architecture of the convnet so far.
model.summary()
# *Adding a classifier on top of the convnet*
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# We’ll do 10-way classification, using a final layer with 10 outputs
and a softmax activation.
# Here’s what the network looks like no
model.summary()
# Define the optimizer
#optimizer = rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
 patience=3,
verbose=1,
factor=0.5,
min_lr=0.00001)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metr
ics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=40)
# Let’s evaluate the model on the test data.
test_loss, test_acc = model.evaluate(X_val, Y_val)
test_acc
results = model.predict(test_data)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
