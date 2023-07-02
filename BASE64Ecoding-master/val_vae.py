import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
#from molecules.model import MoleculeVAE
from molecules.predicted_vae_model import VAE_prop
import os
import h5py
import pickle
import base64
from keras.models import Model,load_model
from rdkit import Chem

# 验证
h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
#data_train = h5f['smiles_train'][:]
#data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
#print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
charset1 = h5f['charset'][:]
charset = []
for i in charset1:
    charset.append(i.decode())
length = len(data_test[0])
#charset = len(data_train[0][0])
h5f.close()
model = VAE_prop()
if os.path.isfile('/data/tp/data/model/predictor_vae_model_250000_12260707.h5'):
    model.load(len(charset), length, '/data/tp/data/model/predictor_vae_model_250000_12260707.h5', latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % '/data/tp/data/model/predictor_vae_model_250000_12260707.h5')
data_test_vae = model.vae_predictor.predict(data_test)[0]

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()
t = 0
tt = 0
ttt = 0
#count  = 0
#count1 = 0
for i in range(5000):
    #item0 = data_test_vae[i].argmax(axis=1)
    #item1 = data_test_vae[m].argmax(axis=1)
    #print(item0)
    # print(item1)
    # break
    #s0 = ''
    item0 = data_test[i].argmax(axis=1)
    s0 = decode_smiles_from_indexes(item0, charset)     
    item = data_test_vae[i].argmax(axis=1)
    s = decode_smiles_from_indexes(item, charset)     
    m = Chem.MolFromSmiles(s)
    if m!= None:
        t+=1
        if s != s0:
            print(s0)
            print(s)
            ttt += 1
#    print(s)
print(t)
print(t/5000)
print(tt)
print(ttt)
