from sklearn import preprocessing
import numpy as np
import pickle
w2v_vector = open('w2v_vector.pkl', 'rb')
w2v_vector = pickle.load(w2v_vector)
data = []
print(w2v_vector['C'])
for k, v in w2v_vector.items():
    data.append(v)
data_normalized = preprocessing.normalize(data, norm='l2')
i = 0
for k in w2v_vector:
    w2v_vector[k] = data_normalized[i]
    i+=1
print(i)
print(w2v_vector['C'])
output_smiles = open('w2v_vector_30_normalized.pkl', 'wb')
pickle.dump(w2v_vector, output_smiles)
output_smiles.close()
