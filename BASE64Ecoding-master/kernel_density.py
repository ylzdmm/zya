import numpy as np
import random
import os
import h5py
#from molecules.predicted_vae_model import VAE_prop
from molecules.predicted_vae_model_w2v import VAE_prop
import matplotlib.pyplot as plt
import seaborn as sns

# 正态分布函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

latent_dim = 196
# 画正态分布图
#h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
data_train = h5f['smiles_train'][:5000]
# data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
logp_test = h5f['logp_test'][:]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
#charset2 = h5f['charset'][:]
#charset1 = []
#for i in charset2:
#    charset1.append(i.decode())
model = VAE_prop()
length = len(data_test[0])
charset = len(data_test[0][0])
#modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
modelname = '/data/tp/data/model/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'

if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

x_latent = model.encoder.predict(data_train)
latent = [[]for _ in range(latent_dim)]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#x_latent = normalization(x_latent)


for i in range(latent_dim):
    for j in range(len(data_train)):
        latent[i].append(x_latent[j][i])
# print(len(latent[0]), numpy.mean(latent[0]), numpy.var(latent[0]), numpy.std(latent[0]))
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
ax.set_xlabel("Z (unstandardized)", fontsize=15)
ax.set_ylabel("Normalized Frequency", fontsize=15)
#ax.set_xlim(-1.0, 1.0)
#ax.set_ylim(0, 30)
#plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], fontsize=15)
#plt.yticks([0, 5, 10, 15, 20, 25, 30], fontsize=15)
for i in range(latent_dim):
    x = latent[i]
    sns.distplot(x, ax=ax, hist=False, kde_kws={"shade": False, "color": (random.random(),
                                                                          random.random(),
                                                                          random.random())})
    # mean = np.mean(latent[i])
    # std = np.std(latent[i])
    # x = np.arange(-1, 1, 0.1)
    # # 设定 y 轴，载入刚才的正态分布函数
    # y = normfun(x, mean, std)
    # plt.plot(x, y, 'b')
#plt.show()
plt.savefig(fname="picture_1/predictor_vae_w2v_kernel(5qed-sas).png",figsize=[6,6],bbox_inches='tight')
