import joblib
import pickle
import h5py
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from molecules.predicted_vae_model_w2v import VAE_prop

#h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
h5f = h5py.File('/data/tp/data/per_all_glove_35_new_w2_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
#logp_train = h5f['logp_train'][:]
#logp_val = h5f['logp_val'][:]
#logp_test = h5f['logp_test'][:]
qed_train = h5f['qed_train'][:]
qed_val = h5f['qed_val'][:]
qed_test = h5f['qed_test'][:]
sas_train = h5f['sas_train'][:]
sas_val = h5f['sas_val'][:]
sas_test = h5f['sas_test'][:]
data = []
logp = []
qed = []
sas = []
for i in range(len(data_train)):
    data.append(data_train[i])
#    logp.append(logp_train[i])
    qed.append(qed_train[i])
    sas.append(sas_train[i])
for j in range(len(data_test)):
    data.append(data_test[j])
#    logp.append(logp_test[j])
    qed.append(qed_test[j])
    sas.append(sas_test[j])
for k in range(len(data_val)):
    data.append(data_val[k])
#    logp.append(logp_val[k])
    qed.append(qed_val[k])
    sas.append(sas_val[k])
data = np.array(data)
#logp = np.array(logp)
qed = np.array(qed)
sas = np.array(sas)
#print(len(data), len(logp), len(qed), len(sas))
target = 5*qed-sas
charset = len(data_train[0][0])
model = VAE_prop()
#modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
#modelname = '/data/tp/data/model/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
modelname = '/data/tp/data/model/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(charset, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

sort = target.argsort()
bottom_2000_index = sort[:2000]
print(target[bottom_2000_index[:10]])

res = []
for i in bottom_2000_index:
    temp = []
    temp.append(i)
    temp.append(target[i])
    temp.append(data[i])
    temp.append(model.encoder.predict(data[i].reshape(1, 120, charset)))
    res.append(temp)

bottom_mol = open('/data/tp/data/bottom_mol_2000_glove(5qed-sas)(std=1).pkl', 'wb')
pickle.dump(res, bottom_mol)
bottom_mol.close()
