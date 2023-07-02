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
from molecules.predicted_vae_model_w2v import VAE_prop
from rdkit import Chem
from scipy.spatial.distance import pdist

model = VAE_prop()
modelname = '/data/tp/data/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

gp = joblib.load('/data/tp/data/model/w2v/Gaussian_model_2000_w2v(5qed-sas)(std=1).pkl')

h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')

w2v_vector = open('data/w2v_vector_35_w2_n1.pkl', 'rb')
w2v_vector = pickle.load(w2v_vector)
word_vector = []
id2word = []
for key in w2v_vector:
    id2word.append(key)
    word_vector.append(w2v_vector[key])

def most_similar(w):
    sims0 = []
    for i in word_vector:
        Y = np.vstack([i, w])
        d0 = pdist(Y, metric='cosine')[0]
        sims0.append(d0)
    sort0 = np.array(sims0).argsort()
    return [(id2word[i], sims0[i]) for i in sort0[:1]]

def decode_smiles_from_vector(vec):
    s = ''
    for j in range(len(vec)):
        s += most_similar(vec[j])[0][0]
    s = s.strip()
    return s

def objective(x):
    return gp.predict([x])[0]*-1

def main():
    data = open('/data/tp/data/bottom_mol_2000_w2v(5qed-sas)(std=1).pkl', 'rb')
    data = pickle.load(data)
    res = []
    for j in range(len(data)):
        temp = {}
        start_index = data[j][0]
        start_latent = data[j][3]
        start_vector = data[j][2]
        start_smiles = decode_smiles_from_vector(start_vector)
        start_target = data[j][1]
        result = minimize(objective, start_latent, method='COBYLA')

        end_latent = result['x']
        s = ''
        end_vector = model.decoder.predict(end_latent.reshape(1, 196))[0]
        end_smiles = decode_smiles_from_vector(end_vector)
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

    optimization = open('/data/tp/data/optimization_result_from_bottom_w2v(5qed-sas)(std=1).pkl', 'wb')
    pickle.dump(res, optimization)

if __name__ == '__main__':
    main()
