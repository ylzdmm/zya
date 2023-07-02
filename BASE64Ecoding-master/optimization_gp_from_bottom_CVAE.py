import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import joblib
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot
from scipy.optimize import minimize
from keras.models import load_model
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem

model = VAE_prop()
modelname = '/data/tp/data/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

gp = joblib.load('/data/tp/data/model/CVAE/Gaussian_model_2000_CVAE(5qed-sas)(std=1).pkl')

h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
charset1 = h5f['charset'][:]
charset = []
for i in charset1:
    charset.append(i.decode())

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def objective(x):
    return gp.predict([x])[0]*-1

def main():
    data = open('/data/tp/data/bottom_mol_2000(5qed-sas)(std=1).pkl', 'rb')
    data = pickle.load(data)
    res = []
    for j in range(len(data)):
        temp = {}
        start_index = data[j][0]
        start_latent = data[j][3]
        start_onehot = data[j][2].argmax(axis=1)
        start_smiles = decode_smiles_from_indexes(start_onehot, charset)
        start_target = data[j][1]
        result = minimize(objective, start_latent, method='COBYLA')

        end_latent = result['x']
        s = ''
        end_onehot = model.decoder.predict(end_latent.reshape(1, 196)).argmax(axis=2)[0]
        end_smiles = decode_smiles_from_indexes(end_onehot, charset)
        end_target = model.predictor.predict(end_latent.reshape(1, 196))[0]
        m = Chem.MolFromSmiles(end_smiles)
        if m != None:
            print('起点：',  start_smiles)
            print('属性值：', start_target)
            print('终点：', end_smiles)
            print('高斯预测属性值：', gp.predict(end_latent)[0])
            print('联合模型预测属性值：', end_target)
            print('有效\n')
            temp['start_index'] = start_index
            temp['start_smiles'] = start_smiles
            temp['start_target'] = start_target
            temp['start_latent'] = start_latent
            temp['end_smiles'] = end_smiles
            temp['end_target'] = end_target
            temp['end_target_gp'] = gp.predict(end_latent)[0]
            temp['end_latent'] = end_latent

    optimization = open('/data/tp/data/optimization_result_from_bottom_CVAE(5qed-sas)(std=1).pkl', 'wb')
    pickle.dump(res, optimization)

if __name__ == '__main__':
    main()
