import numpy as np
import random
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
from molecules.model import MoleculeVAE
import os
import h5py
from molecules.util import base32_vector, vector, vector120, base64_vector, base64_vector_120
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
h5f = h5py.File('data/per_all_42(2).h5', 'r')
smiles_train = h5f['smiles_train'][:]
smiles_val = h5f['smiles_val'][:]
smiles_test = h5f['smiles_test'][:]
logp_train = h5f['logp_train'][:]
logp_val = h5f['logp_val'][:]
logp_test = h5f['logp_test'][:]
qed_train = h5f['qed_train'][:]
qed_val = h5f['qed_val'][:]
qed_test = h5f['qed_test'][:]
sas_train = h5f['sas_train'][:]
sas_val = h5f['sas_val'][:]
sas_test = h5f['sas_test'][:]
charset = h5f['charset'][:]
length = len(smiles_train[0])
# print(charset)
model = MoleculeVAE()
if os.path.isfile('data/vae_model_42(2).h5'):
    model.load(charset, length, 'data/vae_model_42(2).h5', latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % 'data/vae_model_42(2).h5')
smiles_train_latent = model.encoder.predict(smiles_train)
smiles_val_latent = model.encoder.predict(smiles_val)
smiles_test_latent = model.encoder.predict(smiles_test)
print(smiles_train_latent[0])
h5f = h5py.File('data/per_all_latent_42(2).h5', 'w')
h5f.create_dataset('smiles_train_latent', data=smiles_train_latent)
h5f.create_dataset('smiles_val_latent', data=smiles_val_latent)
h5f.create_dataset('smiles_test_latent', data=smiles_test_latent)
h5f.create_dataset('logp_train', data=logp_train)
h5f.create_dataset('logp_val', data=logp_val)
h5f.create_dataset('logp_test', data=logp_test)
h5f.create_dataset('qed_train', data=qed_train)
h5f.create_dataset('qed_val', data=qed_val)
h5f.create_dataset('qed_test', data=qed_test)
h5f.create_dataset('sas_train', data=sas_train)
h5f.create_dataset('sas_val', data=sas_val)
h5f.create_dataset('sas_test', data=sas_test)
h5f.close()
