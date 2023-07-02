import numpy as np
import random
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from molecules.predicted_vae_model_w2v import VAE_prop
from molecules.util import base32_vector, base64_vector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
batch_size = 128
latent_dim = 196
epochs = 1000

def main():
    filename = '/data/tp/data/per_all_w2v_35_w2_n1_250000.h5'
    #filename = '/data/tp/data/per_all_250000.h5'
    h5f = h5py.File(filename, 'r')
    data_train = h5f['smiles_train'][:]
    data_val = h5f['smiles_val'][:]
    logp_train = h5f['logp_train'][:]
    logp_val = h5f['logp_val'][:]
    qed_train = h5f['qed_train'][:]
    qed_val = h5f['qed_val'][:]
    sas_train = h5f['sas_train'][:]
    sas_val = h5f['sas_val'][:]
    target_train = np.array(qed_train)*5 - np.array(sas_train)
    target_val = np.array(qed_val)*5 - np.array(sas_val)
    
    # charset = h5f['charset'][:]
    model = VAE_prop()
    length = len(data_train[0])
    charset = len(data_train[0][0])
    predictorname = '/data/tp/data/model/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
    #predictorname = '/data/tp/data/model/predictor_vae_model_250000_12260707(qed).h5'
    if os.path.isfile(predictorname):
        model.load(charset, length, predictorname, latent_rep_size=latent_dim)
    else:
        model.create(charset, max_length=length, latent_rep_size=latent_dim)

    check_pointer = ModelCheckpoint(filepath=predictorname, verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    TensorBoardname = '/data/tp/data/TensorBoard/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1)'
    #TensorBoardname = '/data/tp/data/TensorBoard/predictor_vae_model_250000_12260707(qed)'

    tbCallBack = TensorBoard(log_dir=TensorBoardname)

    print(data_train[0])
    history = model.vae_predictor.fit(
        data_train,
        [data_train, target_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, tbCallBack],
        validation_data=(data_val, [data_val, target_val])
    )

    model.save(predictorname)

if __name__ == '__main__':
    main()
