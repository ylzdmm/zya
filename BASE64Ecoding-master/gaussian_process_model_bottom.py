import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def score_mae(output, target):
    return sum(abs(output-target))/len(output)


def main():
    #bottom_mol = open('/data/tp/data/bottom_mol_2000(5qed-sas)(std=1).pkl', 'rb')
    #bottom_mol = open('/data/tp/data/bottom_mol_2000_w2v(5qed-sas)(std=1).pkl', 'rb')
    bottom_mol = open('/data/tp/data/bottom_mol_2000_glove(5qed-sas)(std=1).pkl', 'rb')
    bottom_mol = pickle.load(bottom_mol)
    data = []
    target = []
    for i in bottom_mol:
        target.append(i[1])
        data.append(i[3][0])
    #print(data)
    #print(target) 
    model_gp = gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=200)

    model_gp.fit(data, target)

    joblib.dump(model_gp, '/data/tp/data/model/CVAE/Gaussian_model_2000_glove(5qed-sas)(std=1).pkl')

    data_pred = model_gp.predict(data)

    train_score = score_mae(data_pred, target)
    print('训练精度：', train_score)
    # test_pred = model_gp.predict(test_data)
    #
    # test_score = score_mae(test_pred, test_target)
    # print('测试精度：', test_score)


if __name__ == '__main__':
    main()
