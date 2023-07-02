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

from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = VAE_prop()
modelname = '/data/tp/data/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def objective(x):
    return model.predictor.predict(x.reshape(1, 196))[0]*-1

def main():
    h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
    charset1 = h5f['charset'][:]
    data_train = h5f['smiles_train'][:]
    qed_train = h5f['qed_train'][:]
    sas_train = h5f['sas_train'][:]
    target_train = np.array(qed_train) * 5 - np.array(sas_train)
    charset = []
    for i in charset1:
        charset.append(i.decode())
    t = []
    for i in target_train:
        t.append(i)
    t = np.array(t)
    sort = t.argsort()
    bottom_2000 = sort[:5000]
    print(t[bottom_2000[:10]])
    res = []
    data_latent = model.encoder.predict(data_train)

    for j in range(len(bottom_2000)):
        temp = {}
        #start_onehot = data_train[bottom_2000[j]].argmax(axis=1)
        #start_smiles = decode_smiles_from_indexes(start_onehot, charset)
        start_latent = data_latent[bottom_2000[j]]
        #start_pro = t[bottom_2000[j]]
        result = minimize(objective, start_latent, method='COBYLA')
        end_latent = result['x']
        end_onehot = model.decoder.predict(end_latent.reshape(1, 196)).argmax(axis=2)[0]
        end_smiles = decode_smiles_from_indexes(end_onehot, charset)
        m = Chem.MolFromSmiles(end_smiles)
        if m != None:
            start_onehot = data_train[bottom_2000[j]].argmax(axis=1)
            start_smiles = decode_smiles_from_indexes(start_onehot, charset)
            start_pro = t[bottom_2000[j]]
            end_pro = model.predictor.predict(end_latent.reshape(1, 196))[0]
            print('起点：', start_smiles)
            print('起点属性值：', start_pro)
            print('终点：', end_smiles)
            print('终点属性值：', end_pro)
            print('有效\n')
            temp['start_latent'] = start_latent
            temp['start_onehot'] = start_onehot
            temp['start_smiles'] = start_smiles
            temp['start_pro'] = start_pro
            temp['end_latent'] = end_latent
            temp['end_smiles'] = end_smiles
            temp['end_pro'] = end_pro
            res.append(temp)

    print(res, len(res))
    optimization = open('data/optimization_result_from_bottom_CVAE(5qed-sas)(std=1)(5000).pkl', 'wb')
    pickle.dump(res, optimization)
    optimization.close()

if __name__ == '__main__':
    main()
