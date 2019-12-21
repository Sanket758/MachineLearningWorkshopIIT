#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

print(x_train)
print(y_train)

# To change the shape of figure

plt.figure(figsize=(10,10))
for i in range(0,20):

    # 5 Row & 5 Column
    # Plt.subplot split one plot into small plots 

    plt.subplot(5,5, i+1)

    #imshow() it is used with plt.subplot()
    # input x_train 28X28 image of 2D

    plt.imshow(x_train[i] )
    plt.title((y_train[i]))
    plt.xticks([])
    plt.yticks([])

# Define the text labels

fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999

img_index = 70

# y_train contains the lables, ranging from 0 to 9
# y_train[0]

label_index = y_train[img_index]

# Print the label, for example 2 Pullover

print ("y = " + str(label_index) + " " + 
       (fashion_mnist_labels[label_index]))

# # Show one of the images from the training dataset

plt.imshow(x_train[img_index])

#Data normalization
#Normalize the data dimensions so that they are of approximately 
# the same scale. 0-255

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


#Split the training data into training and testing sets 

(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]


# Reshape input data from (28, 28) array to (28, 28, 1) matrix
# reshape() is to reshape array or matrix
# new reshape data will be 784x1

w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

print(x_train)

# One-hot encode the labels
# Apply only over categorical data
# 10 are number of categories of fashion 

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
y_test.shape

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
# 

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, 
                                 padding='same', 
                                 activation='relu', 
                                 input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, 
                                 padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

# Flatten the pixel matrix into feature matrix

model.add(tf.keras.layers.Flatten())

# Input Layer - 16X16

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#Output Layer (Class/Category) - 10

model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

#from keras.callbacks import ModelCheckpoint
#checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid))

# Evaluate the model on test set

score = model.evaluate(x_test, y_test)

# Print test accuracy

print('\n', 'Test accuracy:', score[1])
y_hat = model.predict(x_test)


# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], 
                                        size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i+ 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
    
