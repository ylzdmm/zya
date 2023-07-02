import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

gp = joblib.load('/data/tp/data/model/Gaussian_model_2000_5qed-sas.pkl')

from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem

h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
charset2 = h5f['charset'][:]
charset1 = []
for i in charset2:
    charset1.append(i.decode())
model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_250000_12260707(5qed-sas).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def objective(x):
    # res = []
    # for i in x:
    #     res.append(i)
    return gp.predict([x])[0]*-1

def main():
    latent = open('/data/tp/data/data_train(2000)_latent(5qed-sas).pkl', 'rb')
    latent = pickle.load(latent)
    target = open('/data/tp/data/data_train(2000)_target(5qed-sas).pkl', 'rb')
    target = pickle.load(target)
    t = []
    for i in target:
        t.append(i)
    ind = t.index(max(t))
    ind1 = t.index(min(t))
    print('最大属性值：', t[ind])
    print('最小属性值：', t[ind1])
    res = []
    bounds = ()
    for i in range(196):
        t0 = min(latent[:][i])
        t1 = max(latent[:][i])
        # t0 = -0.1
        # t1 = 0.1
        # b = (-0.1, 0.1)

        # bounds = bounds + ({'type': 'ineq', 'fun': lambda x: x[i] - t0},
        #                    {'type': 'ineq', 'fun': lambda x: -x[i] + t1})
    # print(bounds)
    for j in range(len(latent)):
        temp = []
        pt = np.array(latent[j])
        # bounds = ({'type': 'ineq', 'fun': lambda x: 0.01 - pdist(np.vstack([x, pt]))[0]})
        result = minimize(objective, pt, method='COBYLA')
        solution = result['x']
        # evaluation = objective(solution)
        old = model.decoder.predict(pt.reshape(1, 196)).argmax(axis=2)[0]
        print('起点：', decode_smiles_from_indexes(old, charset1))
        print('属性值：', target[j])


        sampled = model.decoder.predict(solution.reshape(1, 196)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset1)
        print('终点：', sampled)
        print('高斯预测属性值：', gp.predict([solution])[0])
        print('联合模型预测属性值：', model.predictor.predict(solution.reshape(1, 196))[0])
        m = Chem.MolFromSmiles(sampled)
        if m != None:
            print('有效\n')
            temp.append(j)
            temp.append(solution)
            temp.append(gp.predict([solution])[0])
            res.append(temp)
        else:
            print('无效\n')

    print(res, len(res))
    optimization = open('/data/tp/data/optimization_result_joint_model(5qed-sas).pkl', 'wb')
    pickle.dump(res, optimization)
    optimization.close()

    # print(ind2)
    # pca = PCA(n_components=2)
    # latent = pca.fit_transform(latent)
    # print(type(latent))
    # x = latent[:, 0]
    # y = latent[:, 1]
    # print(x)
    # target = objective(qed_test, sas_test)
    # print(target)
    # figure = pyplot.figure()
    # axis = figure.gca(projection='3d')
    # axis.plot_trisurf(x, y, target, cmap='YlGnBu')
    # pyplot.show()



if __name__ == '__main__':
    main()
