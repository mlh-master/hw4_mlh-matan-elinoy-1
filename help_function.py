import os
import numpy as np
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt


def preprocess(datapath):
    # This part reads the images
    classes = ['b', 'c', 'l', 'h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N, num_classes))
    ii = 0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn), 1)
        img = resize(src, (32, 32), order=3)

        images[ii, :, :, 0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii, cc] = 1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b', 'c', 'l', 'h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N, num_classes))
    ii = 0
    for fn in imagelist:

        images[ii, :, :, 0] = imread(os.path.join(datapath, fn), 1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii, cc] = 1
        ii += 1

    return images, Y


def save_model(model, model_name):
    """
    :param model: the ML model to be saved
    :param model_name: (str) the model name
    :return: None
    """
    if not ("results" in os.listdir()):
        os.mkdir("results")
    save_dir = "results/"
    model_name = model_name
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('\nSaved trained model at %s ' % model_path)


def plot_train_stat(history):
    """
    :param history: (tensorflow.python.keras.callbacks.History) contain train and validation statistic
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'val'], loc='upper right')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'val'], loc='upper right')

    plt.tight_layout()
    plt.show()
