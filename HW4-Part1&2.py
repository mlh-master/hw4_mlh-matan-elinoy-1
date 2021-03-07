import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from models import create_new_a_model, create_model_relu, train_and_evaluate_model, create_batch_norm_model, get_net
from help_function import preprocess_train_and_val, preprocess, save_model, plot_train_stat

if __name__ == '__main__':
    """1. setting 
       2. loading  data - creating train, validation and test set  
       3. set all needed parameters
       4. run training and evaluate performance for all net architecture and hyper parameter
            as defined in the HW notebook"""

    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Loading the data for training and validation:
    src_data = '/MLdata/MLcourse/X_ray/'
    train_path = src_data + 'train'
    val_path = src_data + 'validation'
    test_path = src_data + 'test'
    BaseX_train, BaseY_train = preprocess_train_and_val(train_path)
    BaseX_val, BaseY_val = preprocess_train_and_val(val_path)
    X_test, Y_test = preprocess(test_path)

    keras.backend.clear_session()

    # Inputs:
    input_shape = (32, 32, 1)
    learn_rate = 1e-5
    decay = 0
    batch_size = 64
    epochs = 25

    # Define your optimizer parameters:
    AdamOpt = Adam(lr=learn_rate, decay=decay)

    """model relu - Q1"""
    model_relu = create_model_relu(input_shape)
    model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    save_model(model_relu, "model_relu.h5")
    print("\nload model relu for first run")
    model_relu_run_1 = load_model("results/model_relu.h5")
    train_result_relu_1 = train_and_evaluate_model(model_relu_run_1, batch_size, epochs, BaseX_train, BaseY_train,
                                                   X_test, Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(train_result_relu_1)

    """create new_a_model - Q2"""
    new_a_model = create_new_a_model(input_shape)
    new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    save_model(new_a_model, "new_a_model.h5")

    """Number of epochs - Q3"""
    epochs1 = 25
    epochs2 = 40
    print("\nload new_a_model for first run")
    new_model_run_1 = load_model("results/new_a_model.h5")
    new_model_train_result_1 = train_and_evaluate_model(new_model_run_1, batch_size, epochs1, BaseX_train, BaseY_train,
                                                        X_test, Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(new_model_train_result_1)
    print("\nload new_a_model for second run")
    new_model_run_2 = load_model("results/new_a_model.h5")
    new_model_train_result_2 = train_and_evaluate_model(new_model_run_2, batch_size, epochs2, BaseX_train, BaseY_train,
                                                        X_test, Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(new_model_train_result_2)

    """Mini-Batches - Q4"""
    batch_size2 = 32
    epochs3 = 50
    keras.backend.clear_session()
    print("\nload model relu for second run")
    model_relu_run_2 = load_model("results/model_relu.h5")
    train_result_relu_2 = train_and_evaluate_model(model_relu_run_2, batch_size2, epochs3, BaseX_train, BaseY_train,
                                                   X_test, Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(train_result_relu_2)

    """Batch normalization - Q5"""
    AdamOpt = Adam(lr=learn_rate, decay=decay)
    model_batch_norm = create_batch_norm_model(input_shape)
    model_batch_norm.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    print("\nload model relu for second run")
    train_result_model_batch = train_and_evaluate_model(model_batch_norm, batch_size2, epochs3, BaseX_train,
                                                        BaseY_train, X_test, Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(train_result_model_batch)

    """Part 2"""
    input_shape = (32, 32, 1)
    learn_rate = 1e-5
    decay = 1e-03
    batch_size = 64
    epochs = 25
    drop = True
    dropRate = 0.3
    reg = 1e-2
    filters1 = [64, 128, 128, 256, 256]
    NNet1 = get_net(filters1, input_shape, drop, dropRate, reg)

    # Defining the optimizar parameters:
    AdamOpt = Adam(lr=learn_rate, decay=decay)

    # Compile the network:
    NNet1.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

    # Saving checkpoints during training:
    # Checkpath = os.getcwd()

    h1 = NNet1.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0,
                 validation_data=(BaseX_val, BaseY_val), shuffle=True)

    results1 = NNet1.evaluate(X_test, Y_test)
    print('test loss, test acc:', results1)

    """Q1"""
    filters2 = [32, 64, 64, 128, 128]
    NNet2 = get_net(filters2, input_shape, drop, dropRate, reg)

    # Defining the optimizar parameters:
    AdamOpt = Adam(lr=learn_rate, decay=decay)

    # Compile the network:
    NNet2.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

    h2 = NNet2.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0,
                  validation_data=(BaseX_val, BaseY_val), shuffle=True)

    results2 = NNet2.evaluate(X_test, Y_test)
    print('test loss, test acc:', results2)
