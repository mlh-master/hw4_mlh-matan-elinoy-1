# # HW: X-ray images classification
import os
from help_function import preprocess_train_and_val, preprocess, save_model

from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Activation, MaxPooling2D, Permute
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt


# ### PART 1: Fully connected layers ## NN with fully connected layers.
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification.
# Use ReLU activation functions for the hidden layers and He_normal for initialization.
# Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
def create_model_relu(input_shape):
    """

    :param input_shape: (tuple) input shape
    :return: model_relu (Sequential): model relu for Q1 and Q4
    """
    # --------------------------Implement your code here:-------------------------------------
    model_relu = Sequential(name="model_relu")
    model_relu.add(Flatten(input_shape=input_shape))
    model_relu.add(Dense(300, activation='relu', kernel_initializer='he_normal', name='dense_1'))
    model_relu.add(Dense(150, activation='relu',kernel_initializer='he_normal',  name='dense_2'))
    model_relu.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    model_relu.summary()
    return model_relu


# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
def create_new_a_model(input_shape):
    """

    :param input_shape: (tuple) input shape
    :return: new_a_model (Sequential): new model for Q2 and Q3 with than activation function
    (and Xavier normal initializer in accordance) instead of relu
    """
    # --------------------------Implement your code here:-------------------------------------
    new_a_model = Sequential(name="new_a_model")
    new_a_model.add(Flatten(input_shape=input_shape))
    new_a_model.add(Dense(300, activation='tanh', kernel_initializer='glorot_normal', name='dense_1'))
    # new_a_model.add(LeakyReLU())
    new_a_model.add(Dense(150, activation='tanh', kernel_initializer='glorot_normal', name='dense_2'))
    # new_a_model.add(LeakyReLU())
    new_a_model.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    new_a_model.summary()
    return new_a_model


def create_batch_norm_model(input_shape):
    """

    :param input_shape: (tuple) input shape
    :return: new_a_model (Sequential): new_a_model for Q5 added batch normalization after the two dense layer
    """
    # --------------------------Implement your code here:-------------------------------------
    batch_norm_model = Sequential(name="batch_norm_model")
    batch_norm_model.add(Flatten(input_shape=input_shape))
    batch_norm_model.add(Dense(300, activation='tanh', kernel_initializer='glorot_normal', name='dense_1'))
    # batch_norm_model.add(LeakyReLU())
    batch_norm_model.add(BatchNormalization(name='batch_norm_1'))
    batch_norm_model.add(Dense(150, activation='tanh', kernel_initializer='glorot_normal',  name='dense_2'))
    # batch_norm_model.add(LeakyReLU())
    batch_norm_model.add(BatchNormalization(name='batch_norm_2'))
    batch_norm_model.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    batch_norm_model.summary()
    return batch_norm_model


# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task.
# Train your model on the training set and evaluate the model on the testing set. # Print the accuracy and loss over the testing set.
def train_and_evaluate_model(model, batch_size, epochs, BaseX_train, BaseY_train, X_test,
                             Y_test, x_val, Y_val):
    """

    :param model: (Sequential) ML model for training and evaluation
    :param batch_size: (int)
    :param epochs: number of epoch (int)
    :param BaseX_train: (numpy.array) train data
    :param BaseY_train: (numpy.array) train label
    :param X_test: (numpy.array) test data
    :param Y_test: (numpy.array) test label
    :param x_val: (numpy.array) validation data
    :param Y_val: (numpy.array) validation label
    :return: history (tensorflow.python.keras.callbacks.History) contain train and validation statistic
    """
    # --------------------------Implement your code here:-------------------------------------#
    print(f"\nstart training for {epochs} epochs with batch size of {batch_size}")
    history = model.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs
                        ,validation_data=(x_val, Y_val))
    print("Evaluate on test data")
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
    print("Test Accuracy is {:.2f} %".format(100 * loss_and_metrics[1]))

    return history
    # ----------------------------------------------------------------------------------------

def get_net(filters,input_shape,drop,dropRate,reg):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=filters[0], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters[1], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters[2], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters[3], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters[4], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:
    model.add(Dense(512, activation='elu',name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model
