import matplotlib.pyplot as plt
import pickle
import numpy as np
import h5py
from sklearn.decomposition import PCA
from pylab import figure, axes, scatter, title, show
charset = [' ', '-', 'C', '[', '/', '2', '6', '\\', '+', 'n', 'S', '(', 'F', 'c', '3', 'I', '1', ')', 'B', 'r', '#',
           '8', 'o', 'P', 'l', 's', ']', '5', '7', '=', '4', 'O', 'H', '@', 'N']
base64_charset_120 = [' ', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
w2v_vector_30_new_30 = open('data/w2v_vector_30_new.pkl', 'rb')
w2v_vector_30_new_w5 = open('data/w2v_vector_30_new_w5.pkl', 'rb')
w2v_vector_30_new_w10 = open('data/w2v_vector_30_new_w10.pkl', 'rb')
w2v_vector_30_new_w20 = open('data/w2v_vector_30_new_w20.pkl', 'rb')
data0 = pickle.load(w2v_vector_30_new_30)
data1 = pickle.load(w2v_vector_30_new_w5)
data2 = pickle.load(w2v_vector_30_new_w10)
data3 = pickle.load(w2v_vector_30_new_w20)
# print(data0)
# print(data1)
# print(data2)
#s = []
#for key, value in data0.items():
#    s.append(key)
#    print(np.linalg.norm(value))
#print(s, len(s))


x, y, z, p = [], [], [], []
for s in charset:
    x.append(data0[s])
    y.append(data1[s])
    z.append(data2[s])
    p.append(data3[s])
print(np.linalg.norm(x[0]))
print(np.linalg.norm(y[0]))
print(np.linalg.norm(z[0]))
print(np.linalg.norm(p[0]))
#
pca = PCA(n_components=20)
x = pca.fit_transform(x)
y = pca.fit_transform(y)
z = pca.fit_transform(z)
p = pca.fit_transform(p)
#
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)

ax.scatter(x[:, 0], x[:, 1], marker='.', color='r', label=' ', s=1000)
for i in range(0, len(charset)):
    ax.text(x[i, 0], x[i, 1], charset[i], fontsize=15)
show()