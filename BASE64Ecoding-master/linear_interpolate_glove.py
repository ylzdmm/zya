import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import pickle
from molecules.predicted_vae_model_w2v import VAE_prop
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

def main():
    #source = 'CCCC(=O)Nc1ccc(OC[C@H](O)CNC(C)C)c(C(C)=O)c1'
    dest = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'
    source = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'


    #h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
    # data_train = h5f['smiles_train'][:]
    # data_val = h5f['smiles_val'][:]
    #data_test = h5f['smiles_test'][:5000]
    #logp_test = h5f['logp_test'][:5000]
    # print(len(data_train), len(data_val), len(data_test), len(data_train[0]))

    #length = len(data_test[0])
    #charset = len(data_test[0][0])
    #h5f.close()

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


    step = (dest_x_latent - source_x_latent)/float(steps)

    for i in range(steps):
        item = source_x_latent + (step * i)
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
