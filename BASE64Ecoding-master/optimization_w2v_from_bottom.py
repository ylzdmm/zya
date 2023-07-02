import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import joblib
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot
from scipy.optimize import minimize
from keras.models import load_model

gp = joblib.load('/data/tp/data/model/Gaussian_model_w2v_2000_5qed-sas.pkl')
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem

h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_250000.h5', 'r')
model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_707(5qed-sas).h5'
if os.path.isfile(modelname):
    model.load(30, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

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

def objective(x):
    res = []
    for i in x:
        res.append(i)
    return gp.predict([res])[0]*-1

def main():
    # latent = open('/data/tp/data/data_train(2000)_w2v_latent(5qed-sas).pkl', 'rb')
    # latent = pickle.load(latent)
    # target = open('/data/tp/data/data_train(2000)_w2v_target(5qed-sas).pkl', 'rb')
    # target = pickle.load(target)
    data = open('/data/tp/data/bottom_mol_w2v(5qed-sas).pkl', 'rb')
    data = pickle.load(data)

    res = []
    #bounds = ()
    #for i in range(196):
    #    t0 = min(latent[:][i])
    #    t1 = max(latent[:][i])
    #    bounds = bounds + ({'type': 'ineq', 'fun': lambda x: x[i] - t0},
    #                       {'type': 'ineq', 'fun': lambda x: -x[i] + t1})
    #print(bounds)
    for j in range(len(data)):
        temp = []
        pt = data[j][2]
        result = minimize(objective, pt, method='COBYLA')
        # print('Status : %s' % result['message'])
        # print('Total Evaluations: %d' % result['nfev'])
        solution = result['x']
        # evaluation = objective(solution)
        #print('找到的最大属性值：', gp.predict([solution])[0])
        # print(gp.predict([latent[ind]])[0])
        #print(solution)
        # print(latent_to_smiles(solution))
        s = ''
        sampled = model.decoder.predict(solution.reshape(1, 196))[0]
        for k in range(len(sampled)):
            s += most_similar(sampled[k])[0][0]
        sampled = s.strip()
        m = Chem.MolFromSmiles(sampled)
        if m != None:
            print('起点：')
            print('属性值：', data[j][1])
            print('终点：', sampled)
            print('高斯预测属性值：', gp.predict(solution)[0])
            print('联合模型预测属性值：', model.predictor.predict(solution.reshape(1, 196))[0])
            print('有效\n')
            temp.append(j)
            temp.append(solution)
            temp.append(gp.predict([solution])[0])
            res.append(temp)

    optimization = open('/data/tp/data/w2v_vecVAE_optimization_result_from_bottom(5qed-sas).pkl', 'wb')
    pickle.dump(res, optimization)
    optimization.close()
    # solution = np.array(solution)
    # ind2 = np.where(latent == solution)
    #
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
