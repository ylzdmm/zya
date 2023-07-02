from __future__ import print_function
import numpy as np
import random
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import argparse
import os
import h5py
# import matplotlib.pyplot as plt
from molecules.model import MoleculeVAE

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

NUM_EPOCHS = 1000
BATCH_SIZE = 128
LATENT_DIM = 196


def main():
    # args = get_arguments()
    l = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    for i in l:
        filename = 'data/per_all_' + str(i) + '(120)(2).h5'
        # data_train, data_test, charset = load_dataset('data/per_all_25000(index)h5')
        h5f = h5py.File(filename, 'r')
        data_train = h5f['smiles_train'][:]
        data_val = h5f['smiles_val'][:]
        charset = h5f['charset'][:]
        print(len(charset))
        print(charset)
        length = len(data_train[0])
        modelname = 'data/vae_model_' + str(i) + '(120)(3).h5'
        model = MoleculeVAE()
        if os.path.isfile(modelname):
            model.load(charset, modelname, latent_rep_size=LATENT_DIM)
        else:
            model.create(charset, max_length=length, latent_rep_size=LATENT_DIM)

        check_pointer = ModelCheckpoint(filepath=modelname, verbose=1, save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

        TensorBoardname = "TensorBoard/vae_model_" + str(i) + '_120_3'

        tbCallBack = TensorBoard(log_dir=TensorBoardname)

        history = model.autoencoder.fit(
            data_train,
            data_train,
            shuffle=True,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[check_pointer, reduce_lr, early_stopping, tbCallBack],
            validation_data=(data_val, data_val)
        )

if __name__ == '__main__':
    main()
