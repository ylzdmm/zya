import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYTHONHASHSEED'] = 'RANDOM_SEED'
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(RANDOM_SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import h5py
import base64
from functools import reduce
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from molecules.model import MoleculeVAE
from molecules.util import base32_vector, base64_vector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
base32_charset = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base64_charset_120 = ['=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
batch_size = 128
latent_dim = 196
epochs = 1000

def main():
    # l = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # for i in l:
    #     filename = 'data/per_all_base64_' + str(i) + '(120)(2).h5'
    filename = '/data/tp/data/per_all_250000.h5'
    h5f = h5py.File(filename, 'r')
    data_train = h5f['smiles_train'][:]
    data_val = h5f['smiles_val'][:]
    charset = h5f['charset'][:]
    model = MoleculeVAE()
    length = len(data_train[0])
    # modelname = 'vae_model_base64_' + str(i) + '(120)(3).h5'
    modelname = '/data/tp/data/model/vae_model_250000_12260707.h5'
    print(modelname)

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=latent_dim)
    else:
        model.create(charset, max_length=length, latent_rep_size=latent_dim)
    check_pointer = ModelCheckpoint(filepath=modelname, verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    # TensorBoardname = "TensorBoard/vae_model_base64_" + str(i) + '_120_3'

    TensorBoardname = '/data/tp/data/TensorBoard/vae_model_250000_12260707'

    tbCallBack = TensorBoard(log_dir=TensorBoardname)

    print(data_train[0])
    history = model.autoencoder.fit(
        data_train,
        data_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[check_pointer, reduce_lr, early_stopping, tbCallBack],
        validation_data=(data_val, data_val)
    )

if __name__ == '__main__':
    main()
