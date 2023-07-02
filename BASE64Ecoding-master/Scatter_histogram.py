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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')

# 验证
#h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_glove_35_new_w2_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
#logp_train = h5f['logp_train'][:]
#logp_val = h5f['logp_val'][:]
#logp_test = h5f['logp_test'][:]
qed_train = h5f['qed_train'][:]
qed_val = h5f['qed_val'][:]
qed_test = h5f['qed_test'][:]
sas_train = h5f['sas_train'][:]
sas_val = h5f['sas_val'][:]
sas_test = h5f['sas_test'][:]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]

length = len(data_test[0])
charset = len(data_test[0][0])
data = []
logp = []
qed = []
sas = []
for i in range(len(data_train)):
    data.append(data_train[i])
#    logp.append(logp_train[i])
    qed.append(qed_train[i])
    sas.append(sas_train[i])
for j in range(len(data_test)):
    data.append(data_test[j])
#    logp.append(logp_test[j])
    qed.append(qed_test[j])
    sas.append(sas_test[j])
for k in range(len(data_val)):
    data.append(data_val[k])
#    logp.append(logp_val[k])
    qed.append(qed_val[k])
    sas.append(sas_val[k])
data = np.array(data)
#logp = np.array(logp)
qed = np.array(qed)
sas = np.array(sas)
#print(len(data), len(logp), len(qed), len(sas))
target = 5*qed-sas
#print(qed[0], sas[0], target[0])
h5f.close()
model = VAE_prop()

#modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
modelname = '/data/tp/data/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
#modelname = '/data/tp/data/model/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'

if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
x_latent = model.encoder.predict(data)
#pre_out = model.predictor.predict(x_latent)
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

pca = PCA(n_components=50)
x_latent_proj = pca.fit_transform(x_latent)

# tsne = TSNE(n_components=2,
#             random_state=0,
#             verbose=4)
# x_latent_proj = tsne.fit_transform(x_latent)
#print(pca.get_covariance()[0])
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_))
print(sum(pca.explained_variance_ratio_))
del x_latent

x_latent_proj = normalization(x_latent_proj)
x = x_latent_proj[:, 1]
y = x_latent_proj[:, 2]


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
sns.distplot(y, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})

ax2 = fig.add_subplot(spec[0, 1:])
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
a = ax2.scatter(x, y, c=target, cmap='YlGnBu_r', marker='.', s=1)

fig.colorbar(a, ax=ax2)
plt.savefig(fname="picture_1/predictor_vae_w2v_alldata_latent(5qed-sas)(1,2).png",figsize=[5.5,4.5],bbox_inches='tight')
#plt.show()
