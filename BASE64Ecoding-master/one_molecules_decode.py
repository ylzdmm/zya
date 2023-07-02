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
from molecules.util import vector120
from scipy.spatial.distance import pdist

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def main():
    Ibuprofen = 'CC(C)Cc1ccc([C@H](C)C(=O)O)cc1'
    # Ibuprofen = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'

    h5f = h5py.File('data/per_all_250000.h5', 'r')
    data_test = h5f['smiles_test'][:50]
    charset2 = h5f['charset'][:]
    charset1 = []
    for i in charset2:
        charset1.append(i.decode())
    length = len(data_test[0])
    charset = len(data_test[0][0])
    h5f.close()
    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []
    d = []
    modelname = 'model/predictor_vae_model_250000_12260707.h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    Ibuprofen_encoded = vector120(Ibuprofen, charset1)
    Ibuprofen_encoded = np.array(Ibuprofen_encoded)


    Ibuprofen_latent = open('Ibuprofen_latent.pkl', 'rb')
    Ibuprofen_latent = pickle.load(Ibuprofen_latent)
    Ibuprofen_latent = Ibuprofen_latent[0]

    print(Ibuprofen_latent)

    for i in range(5000):
        Ibuprofen_x_latent = model.encoder.predict(Ibuprofen_encoded.reshape(1, width, len(charset1)))

        Ibuprofen_sampling_latent = Ibuprofen_x_latent[0]
        Y = np.vstack([Ibuprofen_latent, Ibuprofen_sampling_latent])
        d0 = pdist(Y)[0]

        sampled = model.decoder.predict(Ibuprofen_x_latent.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset1)
        if sampled == Ibuprofen:
            continue
        m = Chem.MolFromSmiles(sampled)
        if m != None:
            f = 'data/picture/Ibuprofen2/' + str(d0) + '_Ibuprofen.png'
            Draw.MolToFile(m, f, size=(150, 100))
            d.append(d0)
            result.append(sampled)
            print(d0, sampled)
    result1 = list(set(result))
    print(len(result1))
    print(result1)
    print(sum(d)/len(d))
if __name__ == '__main__':
    main()
