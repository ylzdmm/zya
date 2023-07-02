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
from molecules.predicted_vae_model import VAE_prop
from scipy.spatial.distance import pdist
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

optimization_result = open('data/optimization_result_from_bottom_CVAE(5qed-sas)(std=1).pkl', 'rb')
optimization_result = pickle.load(optimization_result)

start = optimization_result[0][0]
end = optimization_result[0][1]
print(start)
print(end)

model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
start = model.decoder.predict(start.reshape(1, 196)).argmax(axis=2)[0]
end = model.decoder.predict(end.reshape(1, 196)).argmax(axis=2)[0]

h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
charset2 = h5f['charset'][:]
charset = []
for i in charset2:
    charset.append(i.decode())
def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

start = decode_smiles_from_indexes(start,charset)
print(start)
m = Chem.MolFromSmiles(start)
f = 'picture_1/start_CVAE.png'
Draw.MolToFile(m, f, size=(150, 100))

end = decode_smiles_from_indexes(end,charset)
#print(end)
m = Chem.MolFromSmiles(end)
f = 'picture_1/end_CVAE.png'
Draw.MolToFile(m, f, size=(150, 100))


