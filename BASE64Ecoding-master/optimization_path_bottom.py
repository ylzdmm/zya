import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
#from molecules.predicted_vae_model import VAE_prop
from molecules.predicted_vae_model_w2v import VAE_prop
#from molecules.predicted_vae_model_glove import VAE_prop
import os
import h5py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# optimization path
#h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_glove_35_new_w2_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
# data_val = h5f['smiles_val'][:]
# data_test = h5f['smiles_test'][:]
qed_train = h5f['qed_train'][:]
sas_train = h5f['sas_train'][:]
# logp_test = h5f['logp_test'][:]
target_train = np.array(qed_train)*5 - np.array(sas_train)
print(max(target_train))

optimization_result = open('data/optimization_result_from_bottom_w2v(5qed-sas)(std=1)(5000).pkl', 'rb')
#optimization_result = open('data/optimization_result_from_bottom_glove(5qed-sas)(std=1)(5000).pkl', 'rb')
optimization_result = pickle.load(optimization_result)

path0 = optimization_result[14]['start_latent']
path1 = optimization_result[14]['end_latent']

length = len(data_train[0])
charset = len(data_train[0][0])

h5f.close()
model = VAE_prop()
#modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
modelname = '/data/tp/data/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
#modelname = '/data/tp/data/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
x_latent = model.encoder.predict(data_train)

print(model.predictor.predict(np.array(path0).reshape(1, 196)), model.predictor.predict(np.array(path1).reshape(1, 196)))

x_latent = x_latent.tolist()
x_latent.append(path0)
x_latent.append(path1)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

pca = PCA(n_components=50)
x_latent_proj = pca.fit_transform(x_latent)

del x_latent

x_latent_proj = normalization(x_latent_proj)
x = x_latent_proj[:, 3]
y = x_latent_proj[:, 4]


fig = plt.figure(figsize=(5.5, 4.5))
spec = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[0.5, 4, 1], height_ratios=[4, 0.5])
spec.update(wspace=0., hspace=0.)
ax = fig.add_subplot(spec[1, 1])

ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel(" ", fontsize=15)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.get_yaxis().set_visible(False)
sns.distplot(x, ax=ax, hist=False, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})

ax1 = fig.add_subplot(spec[0, 0])
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel(" ", fontsize=15)
ax1.set_ylabel("Y", fontsize=15)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.get_xaxis().set_visible(False)
sns.distplot(x, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})

ax2 = fig.add_subplot(spec[0, 1:])
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
a = ax2.scatter(x[:-2], y[:-2], c=target_train, cmap='YlGnBu_r', marker='.', s=1)
ax2.scatter(x[-2], y[-2], marker='*', s=50, color='b')
ax2.scatter(x[-1], y[-1], marker='*', s=50, color='r')
ax2.plot([x[-2], x[-1]], [y[-2], y[-1]], color='r')
fig.colorbar(a, ax=ax2)
plt.savefig(fname="picture_1/optimization_path_w2v(5qed-sas)(std=1)(3,4)(new).png",figsize=[5.5,4.5],bbox_inches='tight')
#plt.show()
