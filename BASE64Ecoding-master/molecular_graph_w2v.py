import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import os
from molecules.predicted_vae_model_w2v import VAE_prop
from scipy.spatial.distance import pdist
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

optimization_result = open('data/optimization_result_from_bottom_w2v(5qed-sas)(std=1).pkl', 'rb')
optimization_result = pickle.load(optimization_result)

start = optimization_result[6][0]
end = optimization_result[6][1]
print(start)
print(end)

model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
start = model.decoder.predict(start.reshape(1, 196))[0]
end = model.decoder.predict(end.reshape(1, 196))[0]

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

start = decode_smiles_from_vector(start)
print(start)
m = Chem.MolFromSmiles(start)
f = 'picture_1/start_w2v.png'
Draw.MolToFile(m, f, size=(150, 100))

end = decode_smiles_from_vector(end)
print(end)
m = Chem.MolFromSmiles(end)
f = 'picture_1/end_w2v.png'
Draw.MolToFile(m, f, size=(150, 100))


