import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.predicted_vae_model_w2v import VAE_prop
import os
import h5py
import pickle
import base64
from keras.models import Model, load_model
from rdkit import Chem
from scipy.spatial.distance import pdist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 验证
h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
# data_train = h5f['smiles_train'][:]
# data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:5000]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]
length = len(data_test[0])
charset = len(data_test[0][0])
h5f.close()
model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
data_test_vae = model.vae_predictor.predict(data_test)[0]

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
        d0 = pdist(Y, metric='Cosine')[0]
        sims0.append(d0)
    sort0 = np.array(sims0).argsort()
    return [(id2word[i], sims0[i]) for i in sort0[:1]]

t = 0
tt = 0
ttt = 0
#count  = 0
#count1 = 0
for i in range(5000):
    s0 = ''
    item0 = data_test[i]
    for n in range(len(item0)):
        # print(item0[n])
        s0 += most_similar(item0[n])[0][0]
    s0 = s0.strip()
#    print(s0)
    for m in range(1):
        item = data_test_vae[i]
#    print(item)
        s = ''
        for j in range(len(item)):
#        print(most_similar(item[j])[0])
            s += most_similar(item[j])[0][0]
        s = s.strip()
#        print(s)
        m = Chem.MolFromSmiles(s)
        if m != None:
            t += 1
            if s != s0:
                # print(s0)
                # print(s)
                ttt += 1
#    print(s)
print(t)
print(t/5000)
print(tt)
print(ttt)
