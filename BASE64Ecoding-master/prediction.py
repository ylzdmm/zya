import numpy
import random
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.models import Sequential
from sklearn.decomposition import PCA
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from functools import reduce
from keras import optimizers
# import matplotlib.pyplot as plt
import h5py

import pickle
import pandas
import zlib
import base64
import struct



latent_rep_size = 196
batch_size = 64
epochs = 1000
max_length = 120
n_topics = 80
steps = 50

# pca = PCA(n_components=80)
# data_train = pca.fit_transform(data_train)
# data_test = pca.fit_transform(data_test)


# ldamodeltrain = lda.LDA(n_topics=n_topics, n_iter=100, random_state=1) #初始化模型, n_iter迭代次数
# ldamodeltrain.fit(data_train)
# data_train = np.array(ldamodeltrain.doc_topic_[:])
# ldamodeltest = lda.LDA(n_topics=n_topics, n_iter=100, random_state=1) #初始化模型, n_iter迭代次数
# ldamodeltest.fit(data_test)
# data_test = np.array(ldamodeltest.doc_topic_[:])
filename = 'data/per_all_250000_1.h5'
h5f = h5py.File(filename, 'r')
# smiles_train = h5f['smiles_train_latent'][:]
# smiles_val = h5f['smiles_val_latent'][:]
# smiles_test = h5f['smiles_test_latent'][:]
smiles_train = h5f['smiles_train'][:]
smiles_val = h5f['smiles_val'][:]
smiles_test = h5f['smiles_test'][:]

for i in ['logp', 'qed', 'sas']:
    pro_train = h5f[i + '_train'][:]
    pro_val = h5f[i + '_val'][:]
    pro_test = h5f[i + '_test'][:]
    # print(smiles_train_latent[0])
    # print(sas_train[0])
    # input_shape = (latent_rep_size,)
    input_shape = (len(smiles_train[0]), len(smiles_train[0][0]))
    # input_shape = (len(smiles_train[0]),)
    modelname = 'data/pro_' + i + 'onehot_model_250000.h5'

    checkpointer = ModelCheckpoint(filepath=modelname, verbose=1, save_best_only=True)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    TensorBoardname = 'TensorBoard/pro_' + i + 'onehot_model_250000'

    tbCallBack = TensorBoard(log_dir=TensorBoardname)
    
    model = Sequential()
    model.add(Flatten(name='flatten_1', input_shape=input_shape))
    for i in range(4):
        model.add(Dense(1024, activation='linear'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(3):
        model.add(Dense(512, activation='linear'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dropout(0.1))
    for i in range(3):
        model.add(Dense(256, activation='linear'))
        model.add(BatchNormalization())
        model.add(Activation('linear'))

    model.add(Dropout(0.3))
    for i in range(2):
        model.add(Dense(64, activation='linear'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Dense(32, activation='linear'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='linear'))
    adam = optimizers.Adam(lr=0.0001)
    model.build((None, 196))
    model.summary()
    model.compile(loss='mean_absolute_error',
                       optimizer=adam, metrics=['mae'])

    # model.compile(loss='mae', optimizer='Adam', metrics=['accuracy'])

    history = model.fit(smiles_train, pro_train,
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpointer, early_stopping, tbCallBack],
                        validation_data=(smiles_val, pro_val))

h5f.close()

# score = model.evaluate(smiles_test_latent, qed_test, verbose=0)
# print('Test loss', score[0])
# print('Test accuracy', score[1])

# Y_predict = model.predict(smiles_test)
# print('property', Y_predict[0], Y_predict[1])

# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = history.epoch
#
# plt.figure(figsize=(8, 8))

# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
