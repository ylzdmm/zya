import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

#gp = joblib.load('/data/tp/data/model/Gaussian_model_2000_5qed-sas.pkl')

#from molecules.predicted_vae_model_w2v import VAE_prop
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
#charset1 = h5f['charset'][:]
data_train = h5f['smiles_train'][:]
qed_train = h5f['qed_train'][:]
sas_train = h5f['sas_train'][:]
target_train = np.array(qed_train) * 5 - np.array(sas_train)
model = VAE_prop()
modelname = '/data/tp/data/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
#modelname = '/data/tp/data/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

x_latent = model.encoder.predict(data_train)
#pca = PCA(n_components=196)
#x_latent = pca.fit_transform(x_latent)

#x_latent = normalization(x_latent)
latent = [[] for _ in range(len(x_latent[0]))]
corr = []
for i in range(len(x_latent[0])):
    for j in range(len(x_latent)):
        latent[i].append(x_latent[j][i])
for i in latent:
    res = np.vstack([i,target_train])
    corr.append(np.corrcoef(res)[0][1])

print(corr)
corr = np.array(corr)
sort = corr.argsort()[::-1]
print(corr[sort[:10]])
#print(max(corr), corr.index(max(corr)))
