import argparse
import numpy as np
import random
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from molecules.model import MoleculeVAE
#import os
import re
import random
import pandas as pd
import struct
import pickle
import h5py
import base64
from rdkit import Chem
import numpy as np
from scipy.spatial.distance import pdist
from functools import reduce
from sklearn.model_selection import train_test_split
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base64_charset_120 = ['=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']


from molecules.util import base32_vector, vector, vector120, base64_vector, base64_vector_120, base32_vector_120, get_w2v_vector, get_w2v_base64_vector
# 按8:1:1划分数据并编码
# smiles = open('smiles(40).pkl', 'rb')
# smiles = pickle.load(smiles)
# print(smiles[0])
# # charset1 = list(reduce(lambda x, y: set(y) | x, smiles, set()))
# # charset1.insert(0, ' ')
# # print(charset1)
# # charset = []
# # for i in charset1:
# #     charset.append(i.encode())
# # print(charset)
# logp = open('logp(40).pkl', 'rb')
# logp = pickle.load(logp)
# print(len(logp))
# qed = open('qed(40).pkl', 'rb')
# qed = pickle.load(qed)
# print(len(qed))
# sas = open('sas(40).pkl', 'rb')
# sas = pickle.load(sas)
# print(len(sas))
# # charset = open('charset.pkl', 'rb')
# # charset = pickle.load(charset)
# # print(len(charset))
# idx = int(len(smiles)/10)
# train_idx = 8*idx
# test_idx = 9*idx
# smiles_train = []
# smiles_val = []
# smiles_test = []
# for s0 in smiles[:train_idx]:
#     smiles_train.append(base64_vector_120(s0))
# for s1 in smiles[train_idx:test_idx]:
#     smiles_val.append(base64_vector_120(s1))
# for s2 in smiles[test_idx:]:
#     smiles_test.append(base64_vector_120(s2))
# h5f = h5py.File('data/per_all_base64_40(120)(2).h5', 'w')
# h5f.create_dataset('smiles_train', data=smiles_train)
# h5f.create_dataset('smiles_val', data=smiles_val)
# h5f.create_dataset('smiles_test', data=smiles_test)
# h5f.create_dataset('logp_train', data=logp[:train_idx])
# h5f.create_dataset('logp_val', data=logp[train_idx:test_idx])
# h5f.create_dataset('logp_test', data=logp[test_idx:])
# h5f.create_dataset('qed_train', data=qed[:train_idx])
# h5f.create_dataset('qed_val', data=qed[train_idx:test_idx])
# h5f.create_dataset('qed_test', data=qed[test_idx:])
# h5f.create_dataset('sas_train', data=sas[:train_idx])
# h5f.create_dataset('sas_val', data=sas[train_idx:test_idx])
# h5f.create_dataset('sas_test', data=sas[test_idx:])
# # h5f.create_dataset('charset', data=charset)
# h5f.close()





# 数据编码
def data_encoder(t, train_idx, val_idx, test_idx):
    w2v_vector = open('data/glove_vector_35_w2_new.pkl', 'rb')
    w2v_vector = pickle.load(w2v_vector)

    smiles_train = []
    smiles_val = []
    smiles_test = []
    if t == 0:
        #charset1 = list(reduce(lambda x, y: set(y) | x, smiles, set()))
        #charset1.insert(0, ' ')
        #print(charset1)
        #charset = []
        #for i in charset1:
        #    charset.append(i.encode())
        #print(charset)
        for s0 in smiles[train_idx]:
            smiles_train.append(get_w2v_vector(s0, w2v_vector))
        for s1 in smiles[val_idx]:
            smiles_val.append(get_w2v_vector(s1, w2v_vector))
        for s2 in smiles[test_idx]:
            smiles_test.append(get_w2v_vector(s2, w2v_vector))
        print(smiles_test[0][0])
        h5f = h5py.File('/data/tp/data/per_all_glove_35_w2_new_250000.h5', 'w')
        h5f.create_dataset('smiles_train', data=smiles_train)
        h5f.create_dataset('smiles_val', data=smiles_val)
        h5f.create_dataset('smiles_test', data=smiles_test)
        h5f.create_dataset('logp_train', data=logp[train_idx])
        h5f.create_dataset('logp_val', data=logp[val_idx])
        h5f.create_dataset('logp_test', data=logp[test_idx])
        h5f.create_dataset('qed_train', data=qed[train_idx])
        h5f.create_dataset('qed_val', data=qed[val_idx])
        h5f.create_dataset('qed_test', data=qed[test_idx])
        h5f.create_dataset('sas_train', data=sas[train_idx])
        h5f.create_dataset('sas_val', data=sas[val_idx])
        h5f.create_dataset('sas_test', data=sas[test_idx])
        #h5f.create_dataset('charset', data=charset)
        h5f.close()
    if t == 1:
        for s0 in smiles[train_idx]:
            smiles_train.append(base64_vector_120(s0))
        for s1 in smiles[val_idx]:
            smiles_val.append(base64_vector_120(s1))
        for s2 in smiles[test_idx]:
            smiles_test.append(base64_vector_120(s2))
        print(smiles_test[0][0])
        h5f = h5py.File('/data/tp/data/per_all_base64_250000.h5', 'w')
        h5f.create_dataset('smiles_train', data=smiles_train)
        h5f.create_dataset('smiles_val', data=smiles_val)
        h5f.create_dataset('smiles_test', data=smiles_test)
        h5f.create_dataset('logp_train', data=logp[train_idx])
        h5f.create_dataset('logp_val', data=logp[val_idx])
        h5f.create_dataset('logp_test', data=logp[test_idx])
        h5f.create_dataset('qed_train', data=qed[train_idx])
        h5f.create_dataset('qed_val', data=qed[val_idx])
        h5f.create_dataset('qed_test', data=qed[test_idx])
        h5f.create_dataset('sas_train', data=sas[train_idx])
        h5f.create_dataset('sas_val', data=sas[val_idx])
        h5f.create_dataset('sas_test', data=sas[test_idx])
        h5f.close()
    if t == 2:
        for s0 in smiles[train_idx]:
            smiles_train.append(base32_vector_120(s0))
        for s1 in smiles[val_idx]:
            smiles_val.append(base32_vector_120(s1))
        for s2 in smiles[test_idx]:
            smiles_test.append(base32_vector_120(s2))
        print(smiles_test[0][0])
        h5f = h5py.File('/data/tp/data/per_all_base32_250000.h5', 'w')
        h5f.create_dataset('smiles_train', data=smiles_train)
        h5f.create_dataset('smiles_val', data=smiles_val)
        h5f.create_dataset('smiles_test', data=smiles_test)
        h5f.create_dataset('logp_train', data=logp[train_idx])
        h5f.create_dataset('logp_val', data=logp[val_idx])
        h5f.create_dataset('logp_test', data=logp[test_idx])
        h5f.create_dataset('qed_train', data=qed[train_idx])
        h5f.create_dataset('qed_val', data=qed[val_idx])
        h5f.create_dataset('qed_test', data=qed[test_idx])
        h5f.create_dataset('sas_train', data=sas[train_idx])
        h5f.create_dataset('sas_val', data=sas[val_idx])
        h5f.create_dataset('sas_test', data=sas[test_idx])
        h5f.close()
data = pd.read_hdf('/data/tp/data/zinc-1.h5', 'table')
smiles = data['smiles']
logp = data['logp']
qed = data['qed']
sas = data['sas']
train_idx, val_test_idx = map(np.array, train_test_split(smiles.index, test_size = 0.20))
val_idx, test_idx = map(np.array, train_test_split(val_test_idx, test_size = 0.50))
print(len(train_idx), len(val_test_idx), len(val_idx), len(test_idx))
for i in range(1):
    data_encoder(i, train_idx, val_idx, test_idx)


# 平均欧式距离
# h5f = h5py.File('data/per_all_latent_test.h5', 'r')
# smiles_train_latent = h5f['smiles_train_latent'][:]
# print(len(smiles_train_latent))
# distances = []
# for i in range(10000):
#     x = random.randint(0, 10000)
#     source = smiles_train_latent[x]
#     y = random.randint(10000, 20000)
#     dest = smiles_train_latent[y]
#     Y = np.vstack([source, dest])  # 将x,y两个一维数组合并成一个2D数组 ；[[x1,x2,x3...],[y1,y2,y3...]]
#     distance = pdist(Y)
#     print(distance)
#     distances.append(distance)
# print(np.mean(distances))


# 取固定长度字符
# smiles_data = []
# logp_data = []
# qed_data = []
# sas_data = []
# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# for index, value in enumerate(smiles):
#     if len(value) == 47:
#         smiles_data.append(value)
#         logp_data.append(logp[index])
#         qed_data.append(qed[index])
#         sas_data.append(sas[index])
# print(len(smiles_data))
# output_smiles = open('smiles(47).pkl', 'wb')
# output_logp = open('logp(47).pkl', 'wb')
# output_qed = open('qed(47).pkl', 'wb')
# output_sas = open('sas(47).pkl', 'wb')
# pickle.dump(smiles_data, output_smiles)
# pickle.dump(logp_data, output_logp)
# pickle.dump(qed_data, output_qed)
# pickle.dump(sas_data, output_sas)
# output_smiles.close()
# output_logp.close()
# output_qed.close()
# output_sas.close()





