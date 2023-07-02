import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from molecules.predicted_vae_model import VAE_prop
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def score_mae(output, target):
    return sum(abs(output-target))/len(output)


def main():

    h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
    #h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_250000.h5', 'r')
    #h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_250000.h5', 'r')
    # data_train = h5f['smiles_train'][:]
    # data_val = h5f['smiles_val'][:]
    data_train = h5f['smiles_train'][:2000]
    logp_train = h5f['logp_train'][:2000]
    qed_train = h5f['qed_train'][:2000]
    sas_train = h5f['sas_train'][:2000]
    target_train = np.array(qed_train)*5 - np.array(sas_train)
    print(qed_train[0], sas_train[0], target_train[0])

    length = len(data_train[0])
    charset = len(data_train[0][0])
    h5f.close()

    model = VAE_prop()

    #modelname = '/data/tp/data/model/predictor_vae_model_250000_12260707(5qed-sas).h5'
    modelname = '/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_707(5qed-sas).h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)
    data_train_vae = model.encoder.predict(data_train)

    latent = open('/data/tp/data/data_train(2000)_w2v_latent(5qed-sas).pkl', 'wb')
    pickle.dump(data_train_vae, latent)
    latent.close()

    target = open('/data/tp/data/data_train(2000)_w2v_target(5qed-sas).pkl', 'wb')
    pickle.dump(target_train, target)
    target.close()

    train_data = data_train_vae[:]
    train_target = target_train[:]
    # test_data = data_train_vae[:]
    # test_target = target_train[:]
    model_gp = gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=200)

    model_gp.fit(train_data, train_target)

    joblib.dump(model_gp, '/data/tp/data/model/Gaussian_model_w2v_2000_5qed-sas.pkl')

    train_pred = model_gp.predict(train_data)

    train_score = score_mae(train_pred, train_target)
    print('训练精度：', train_score)
    # test_pred = model_gp.predict(test_data)
    #
    # test_score = score_mae(test_pred, test_target)
    # print('测试精度：', test_score)


if __name__ == '__main__':
    main()
