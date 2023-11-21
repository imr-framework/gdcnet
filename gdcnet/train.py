import os

import tensorflow as tf
import keras.losses
import numpy as np

from gdcnet.data.utils import *
from gdcnet.model.gdcnet_vm import gdcnet_vm
from gdcnet.model.losses import NCC, Grad

def train_model(data_dir, model_dir, train_type, ndims):

    # Model parameters
    batch_size = 1
    epochs = 100
    lr = 1e-5
    if train_type == 'sup':
        network_arch = 'unet'
    elif train_type == 'semisup' or train_type == 'selfsup':
        network_arch = 'unet+stu'
    else:
        raise ValueError('train_type must be sup, semisup or selfsup')

    # Load the training data
    x_train, y_train = load_data_from_txt(data_dir, os.path.join(model_dir, 'train.txt'), ndims)
    x_val, y_val = load_data_from_txt(data_dir, os.path.join(model_dir, 'val.txt'), ndims)

    # Input shape
    inshape = x_train[0].shape[:-1]

    # Build the model
    model = gdcnet_vm(inshape=inshape, arch=network_arch)

    # Loss functions
    sim_loss_vdm = keras.losses.MeanSquaredError()
    sim_loss_T1wEPIdc = NCC().loss
    smooth_loss = Grad('l2', loss_mult=2).loss

    inputs = [np.expand_dims(x_train[..., 0], axis=-1), np.expand_dims(x_train[..., 1], axis=-1)] # [EPId, T1w]
    if train_type == 'sup':
        outputs = [np.expand_dims(y_train[...,0], axis=-1)] # VDM
        losses = sim_loss_vdm
        val_data = ([np.expand_dims(x_val[..., 0], axis=-1), np.expand_dims(x_val[..., 1], axis=-1)], np.expand_dims(y_val[...,0], axis=-1))

    elif train_type == 'semisup':
        outputs = [np.expand_dims(x_train[..., 1], axis=-1), np.expand_dims(y_train[...,0], axis=-1)] # [T1w, VDM]
        losses = [sim_loss_T1wEPIdc, sim_loss_vdm]
        val_data = ([np.expand_dims(x_val[..., 0], axis=-1), np.expand_dims(x_val[..., 1], axis=-1)], [np.expand_dims(x_val[..., 1], axis=-1), np.expand_dims(y_val[...,0], axis=-1)])

    elif train_type == 'selfsup':
        # Combine training and validation data
        x_train = np.concatenate((x_train, x_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        # Randomly shuffle the data
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        inputs = [np.expand_dims(x_train[..., 0], axis=-1), np.expand_dims(x_train[..., 1], axis=-1)]  # [EPId, T1w]
        outputs = [np.expand_dims(x_train[..., 1], axis=-1), np.expand_dims(y_train[...,0], axis=-1)] # [T1w, VDM]
        losses = [sim_loss_T1wEPIdc, smooth_loss]
        val_data = None

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=losses)
    model.summary()

    # Train the model
    history = model.fit(inputs,
                        outputs,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=val_data)

    # Save the model
    model.save(model_dir + f'/model_{ndims}D_{train_type}.h5')

    # Save the training history
    np.save(os.path.join(model_dir, f'history_{ndims}D_{train_type}'), history.history)

if __name__ == "__main__":
    data_dir = '../data/preprocessed/train_testID'
    ndims = [2, 2, 2, 3, 3, 3]
    train_type = ['sup', 'semisup', 'selfsup', 'sup', 'semisup', 'selfsup']
    # Check that the number of training types and dimensions are the same
    # raise error if not
    if not len(ndims) == len(train_type):
        raise ValueError('The number of training types and dimensions must be the same')
    model_dir = '../models/20231116'
    for i in range(len(ndims)):
        train_model(data_dir, model_dir, train_type[i], ndims[i])