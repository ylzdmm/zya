import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import pickle
from molecules.predicted_vae_model_glove import VAE_prop
from rdkit import Chem
from rdkit.Chem import Draw
from molecules.util import vector120, get_w2v_vector
from keras.models import Model, load_model
from scipy.spatial.distance import pdist

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
w2v_vector = open('data/glove_vector_35_new_w2.pkl', 'rb')
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


def slerp(val, low, high):
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    # 求反余弦arccos   np.dot求矩阵乘法得余弦值   np.linalg.norm求范数
    # 得到向量夹角
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    # 求正弦
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def main():
    #source = 'CCCC(=O)Nc1ccc(OC[C@H](O)CNC(C)C)c(C(C)=O)c1'
    source = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    #source = 'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1'
    dest = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'

    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []

    modelname = '/data/tp/data/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
   
    if os.path.isfile(modelname):
        model.load(35, 120, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    source_encoded = get_w2v_vector(source, w2v_vector)
    dest_encoded = get_w2v_vector(dest, w2v_vector)
    source_encoded = np.array(source_encoded)
    dest_encoded = np.array(dest_encoded)
    source_x_latent = model.encoder.predict(source_encoded.reshape(1, width, 35))
    dest_x_latent = model.encoder.predict(dest_encoded.reshape(1, width, 35))
    source_x_latent = source_x_latent[0]
    dest_x_latent = dest_x_latent[0]

    for i in range(steps):
        item = slerp(i/steps, source_x_latent, dest_x_latent)
        sampled = model.decoder.predict(item.reshape(1, latent_dim))[0]
        s = decode_smiles_from_vector(sampled)
        m = Chem.MolFromSmiles(s)
        if m != None:
            if s not in result:
                result.append(s)
            print(s)
    #result1 = list(set(result))
    print(len(result))
    print(result)
if __name__ == '__main__':
    main()
