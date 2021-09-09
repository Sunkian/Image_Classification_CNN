import os
import time
import numpy as np
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.layers import Activation,BatchNormalization, Conv2D, MaxPooling2D, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, mean_absolute_error


# Variables
jpgs = []
pngs = []
resultX = []
resultY = []
dataset = 'challenge_dataset/'
stored_jpgs = 'Stored-jpgs/'
stored_pngs = 'Stored-pngs/'
stored_npys = 'Stored-npys/'
train = pd.read_csv('challenge_dataset/labels.txt', sep=' ', names=['id_image', 'label'])

# Names of the folders
# Stored_jpgs/pngs/npys : Folders where the images with their respective format are stored
# CONCAT : Folders that contains the concatenation of the three images in a numpy array format

### How to know how many images there are in each class :
# df = pd.read_csv('challenge_dataset/labels.txt', sep=' ', names=['id_image', 'id_class'])
# df['id_class'].value_counts()
### --->
# 0    71
# 3    31
# 2    25
# 1    22


def separate(dataset, jpg_out, png_out, npy_out):
    """
    This function separates in three different folders the images
    with different extensions (.jpg, .png, .npy) and converts more
    specifically the .jpg and .png to .npy files.
    :param dataset: The main dataset where all the images + .txt files are stored.
    :param jpg_out: The directory where we will store all the .jpg images. (Grayscale)
    :param png_out: The directory where we will store all the .png images. (Infra-Red)
    :param npy_out: The directory where we will store all the .npy images. (Depth)
    """
    for filename in sorted(os.listdir(dataset)):
        if filename.endswith('.jpg'):
            imgs = Image.open(dataset + filename)
            imgs_arrays = np.asarray(imgs)
            jpgs.append(imgs_arrays)
            names = (os.path.splitext(filename)[0])
            np.save(jpg_out + names + '.npy', imgs_arrays)

        elif filename.endswith('.png'):
            imgs = Image.open(dataset + filename)
            imgs_arrays = np.asarray(imgs)
            pngs.append(imgs_arrays)
            names = (os.path.splitext(filename)[0])
            np.save(png_out + names + '.npy', imgs_arrays)

        if filename.endswith('.npy'):
            imgs = np.load(dataset + filename)
            np.save(npy_out + filename, imgs)


separate(dataset, stored_jpgs, stored_pngs, stored_npys)


def convert_and_concat(jpg_in, png_in, npy_in):
    """

    :param jpg_in:
    :param png_in:
    :param npy_in:
    :return:
    """
    for i, j, k in zip(os.listdir(jpg_in), os.listdir(png_in), os.listdir(npy_in)):
        data_i = np.load(jpg_in + i)
        data_j = np.load(png_in + j)
        data_k = np.load(npy_in + k)
        name = (os.path.splitext(i)[0])
        # print(data_i.shape)
        # print(data_j.shape)
        # print(data_k.shape)

        if i[0] == j[0] == k[0]:
            big = np.concatenate((data_i, data_j, data_k[...,None]), axis=2) # the axis along which you concatenate : If none, arrays are flattened, 1 (horizontally), 0 (vertically), 2 (3rd dimension)
            #print(big.shape) # (480, 848, 7)
            np.save('CONCAT/' + name + '.npy', big)


convert_and_concat(stored_jpgs, stored_pngs, stored_npys)


def process_labels(id_img, id_label):
    """

    :param id_img:
    :param id_class:
    :return:
    """
    with open("challenge_dataset/labels.txt") as fp:
        lines = fp.readlines()
        for line in lines:
            id_img.append(line.split(' ')[0])
            id_label.append(line.split(' ')[1].strip('\n'))
            a = line.split(' ')[0]


process_labels(resultX, resultY)

train_image = []
train_names = []
for i in tqdm(range(train.shape[0])):
    img = np.load('CONCAT/' + resultX[i] + '.npy')
    # print(img.shape)
    train_image.append(img)
    train_names.append(resultX[i])

X = np.array(train_image)
y = resultY
y = np.array(y)
y = to_categorical(y)

train_ratio = 0.8
validation_ratio = 0.10
test_ratio = 0.10

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))


print("X_train shape: {}".format(X_train.shape)) # X_train shape: (119, 480, 848, 7)
print("X_test shape: {}".format(X_test.shape)) # X_test shape: (15, 480, 848, 7)
print("X_val shape: {}".format(X_val.shape)) # X_val shape: (15, 480, 848, 7)
print("y_train shape: {}".format(y_train.shape)) # y_train shape: (119, 4)
print("y_test shape: {}".format(y_test.shape)) # y_test shape: (15, 4)
print("y val shape: {}".format(y_val.shape)) # y val shape: (15, 4)

# VGGNet 16
custom_vgg = Sequential()
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (480, 848, 7)))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Flatten())
custom_vgg.add(Dense(4, activation = "softmax"))

custom_vgg.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

custom_vgg.summary()

opt = Adam(lr=0.000001) # smoother curve
custom_vgg.compile(optimizer = opt , loss = tensorflow.keras.losses.categorical_crossentropy, metrics = ['accuracy'])

t0 = time.time()

histo = custom_vgg.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

print("training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

# # Accuracy
prediction = custom_vgg.predict(X_test)
print('ALL prediction : ', prediction)
print('1st pred: ', prediction[0])
print('label prediction of the 1st image in the test set : ', np.argmax(prediction[0]))
classes_prediction = np.argmax(prediction, axis=1)
classes_prediction2 = np.round(prediction).astype(int)
print('Class prediction : ', classes_prediction)
print('Class prediction2 : ', classes_prediction2)



acc = histo.history['accuracy']
val_acc = histo.history['val_accuracy']
loss = histo.history['loss']
val_loss = histo.history['val_loss']
epochs_range = range(20)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('test.png')
plt.show()

# Simple cnn
# model = Sequential()  #allows you to build a model layer by layer
# model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(480, 848, 7)))
# model.add(MaxPool2D())
#
# model.add(Conv2D(32, 3, padding="same", activation="relu"))
# model.add(MaxPool2D())
#
# model.add(Conv2D(64, 3, padding="same", activation="relu"))
# model.add(MaxPool2D())
# model.add(Dropout(0.4)) # Avoid overfitting
#
# model.add(Flatten())
# model.add(Dense(128,activation="relu"))
# model.add(Dense(4, activation="softmax"))
#
# model.summary()
#
# opt = Adam(lr=0.000001) # smoother curve
# model.compile(optimizer = opt , loss = tensorflow.keras.losses.categorical_crossentropy, metrics = ['accuracy'])
