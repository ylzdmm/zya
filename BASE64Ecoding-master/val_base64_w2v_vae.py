import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.w2v_model import MoleculeVAE
import os
import h5py
import pickle
import base64
from keras.models import Model,load_model
from rdkit import Chem
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base64_charset_120 = ['=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
base32_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
# 验证
h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_base64_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]
length = len(data_train[0])
charset = len(data_train[0][0])
h5f.close()
model = MoleculeVAE()
if os.path.isfile('/data/tp/data/model/vae_model_w2v_30_new_base64_250000_42.h5'):
    model.load(charset, length, '/data/tp/data/model/vae_model_w2v_30_new_base64_250000_42.h5', latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % '/data/tp/data/model/vae_model_w2v_30_new_base64_250000_42.h5')
data_test_vae = model.autoencoder.predict(data_test)

#word2id = open('word2id.pkl', 'rb')
#id2word = open('id2word.pkl', 'rb')

word2id = open('word2id_base64.pkl', 'rb')
id2word = open('id2word_base64.pkl', 'rb')

word2id = pickle.load(word2id)
id2word = pickle.load(id2word)


model = load_model('./word2vec_base64.h5')
embeddings = model.get_weights()[0]
normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5

def most_similar(w):
    sims = np.dot(normalized_embeddings, w)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i],sims[i]) for i in sort[:1]]
    #return v
w2v_vector = open('w2v_vector_30_new_base64.pkl', 'rb')
w2v_vector = pickle.load(w2v_vector)

t = 0
tt = 0
ttt = 0
#count  = 0
#count1 = 0
for i in range(5000):
    #item0 = data_test[m].argmax(axis=1)
    #item1 = data_test_vae[m].argmax(axis=1)
    #print(item0)
    # print(item1)
    # break
    s0 = ''
    item0 = data_test[i]
    for n in range(len(item0)):
        s0 += most_similar(item0[n])[0][0]     
    s0 = s0.strip()
    s0 = base64.b64decode(s0).decode() 
    #print(s0)
    m0 = Chem.MolFromSmiles(s0)
    for m in range(1):
        item = data_test_vae[i]
#    print(item)
        s = ''
        for j in range(len(item)):
#        print(most_similar(item[j])[0])
            s+=most_similar(item[j])[0][0]
        s = s.strip()
        try:
            s = base64.b64decode(s).decode()
        except:
            tt += 0
            continue
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
        # 补空格时用于跳出循环
        # if item0[j] == 0:
          #   break
#        count0 += 1
#        if item0[j] != item1[j]:
#            count1 += 1
#print(count0, count1)
#print((count0-count1)/count0)
