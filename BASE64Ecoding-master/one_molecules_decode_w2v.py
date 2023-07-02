import numpy as np
import random
# RANDOM_SEED = 12260707
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
import h5py
import os
import pickle
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
from rdkit.Chem import Draw
from molecules.util import vector120, get_w2v_vector
from scipy.spatial.distance import pdist
from keras.models import Model, load_model

def main():
    Ibuprofen = 'CC(C)Cc1ccc([C@H](C)C(=O)O)cc1'
    # Ibuprofen = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'

    w2v_vector = open('data/w2v_vector_30_new.pkl', 'rb')
    w2v_vector = pickle.load(w2v_vector)

    h5f = h5py.File('data/per_all_w2v_30_new_250000.h5', 'r')
    data_test = h5f['smiles_test'][:50]

    w2v = load_model('model/word2vec.h5')
    embeddings = w2v.get_weights()[0]
    normalized_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5

    word2id = open('data/word2id.pkl', 'rb')
    id2word = open('data/id2word.pkl', 'rb')
    word2id = pickle.load(word2id)
    id2word = pickle.load(id2word)

    def most_similar(w):
        sims = np.dot(normalized_embeddings, w)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(id2word[i], sims[i]) for i in sort[:1]]
        # return v
    length = len(data_test[0])
    charset = len(data_test[0][0])
    h5f.close()
    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []
    d = []
    modelname = 'model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    Ibuprofen_encoded = get_w2v_vector(Ibuprofen, w2v_vector)
    Ibuprofen_encoded = np.array(Ibuprofen_encoded)

    Ibuprofen_latent = open('Ibuprofen_latent_w2v.pkl', 'rb')
    Ibuprofen_latent = pickle.load(Ibuprofen_latent)


    print(Ibuprofen_latent)

    for i in range(2000):
        Ibuprofen_x_latent = model.encoder.predict(Ibuprofen_encoded.reshape(1, width, charset))

        Ibuprofen_sampling_latent = Ibuprofen_x_latent[0]
        Y = np.vstack([Ibuprofen_latent, Ibuprofen_sampling_latent])
        d0 = pdist(Y)[0]

        sampled = model.decoder.predict(Ibuprofen_x_latent.reshape(1, latent_dim))[0]
        s = ''
        for j in range(len(sampled)):
            s += most_similar(sampled[j])[0][0]
        s = s.strip()
        if s == Ibuprofen:
            continue
        m = Chem.MolFromSmiles(s)
        if m != None:
            f = 'data/picture/Ibuprofen_w2v/' + str(d0) + '_Ibuprofen.png'
            Draw.MolToFile(m, f, size=(150, 100))
            d.append(d0)
            result.append(s)
            print(d0, s)
    result1 = list(set(result))
    print(len(result1))
    print(result1)
    print(sum(d)/len(d))
if __name__ == '__main__':
    main()
