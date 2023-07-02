import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.predicted_vae_model import VAE_prop
import os
import h5py
import pickle
import base64
from keras.models import Model,load_model
from rdkit import Chem
from pylab import figure, axes, scatter, title, show
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# 验证
h5f = h5py.File('data/per_all_250000.h5', 'r')
# data_train = h5f['smiles_train'][:]
# data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:100]
logp_test = h5f['logp_test'][:100]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]
length = len(data_test[0])
charset = len(data_test[0][0])
h5f.close()
model = VAE_prop()

modelname = 'model/predictor_vae_model_250000_12260707.h5'

if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

x_latent = model.encoder.predict(data_test)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# pca = PCA(n_components=50)
# x_latent = pca.fit_transform(x_latent)
#
# figure(figsize=(6, 6))
# scatter(x_latent[:, 0], x_latent[:, 1], marker='.')
# show()

# tsne = TSNE(n_components=2,
#             perplexity=30.0,
#             learning_rate=750.0,
#             n_iter=1000,
#             verbose=4)
tsne = TSNE(n_components=2,
            random_state=0,
            verbose=4)
x_latent_proj = tsne.fit_transform(x_latent)
del x_latent
x_latent_proj = normalization(x_latent_proj)

fig = plt.figure(figsize=(6, 6))


ax = fig.add_subplot()
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.scatter(x_latent_proj[:, 0], x_latent_proj[:, 1], c=logp_test, cmap='YlGnBu', marker='.')
plt.colorbar()
plt.show()
