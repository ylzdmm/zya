import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.predicted_vae_model import VAE_prop
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 验证
h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
logp_train = h5f['logp_train'][:]
logp_val = h5f['logp_val'][:]
logp_test = h5f['logp_test'][:]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]

length = len(data_test[0])
charset = len(data_test[0][0])
data = []
logp = []
for i in range(len(data_train)):
    data.append(data_train[i])
    logp.append(logp_train[i])
#for j in range(len(data_test)):
#    data.append(data_test[j])
#    logp.append(logp_test[j])
#for k in range(len(data_val)):
#    data.append(data_val[k])
#    logp.append(logp_val[k])
data = np.array(data)
logp = np.array(logp)
print(len(data))
# data_test = data_train + data_val + data_test
# logp_test = logp_train + logp_val + logp_test
# print(len(data_test), len(logp_test))

h5f.close()
model = VAE_prop()

modelname = '/data/tp/data/model/predictor_vae_model_250000_12260707(5qed-sas).h5'
#modelname = '/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5'

if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
x_latent = model.encoder.predict(data)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

pca = PCA(n_components=2)
x_latent_proj = pca.fit_transform(x_latent)

# tsne = TSNE(n_components=2,
#             random_state=0,
#             verbose=4)
# x_latent_proj = tsne.fit_transform(x_latent)

del x_latent

x_latent_proj = normalization(x_latent_proj)
x = x_latent_proj[:, 0]
y = x_latent_proj[:, 1]


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
a = ax2.scatter(x, y, c=logp, cmap='YlGnBu', marker='.', s=1)

fig.colorbar(a, ax=ax2)

plt.savefig(fname="predictoe_vae_alldata_latent(5qed-sas).png",figsize=[5.5,4.5])
plt.show()
