import matplotlib.pyplot as plt
import pickle
import numpy as np
import h5py
from sklearn.decomposition import PCA
from pylab import figure, axes, scatter, title, show
from sklearn.manifold import TSNE

charset = [' ', '-', 'C', '[', '/', '2', '6', '\\', '+', 'n', 'S', '(', 'F', 'c', '3', 'I', '1', ')', 'B', 'r', '#',
           '8', 'o', 'P', 'l', 's', ']', '5', '7', '=', '4', 'O', 'H', '@', 'N']
base64_charset_120 = [' ', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
w2v_vector_30_new_30 = open('data/w2v_vector_30_new.pkl', 'rb')
# w2v_vector_30_new_w5 = open('data/w2v_vector_30_new_w5.pkl', 'rb')
# w2v_vector_30_new_w10 = open('data/w2v_vector_30_new_w10.pkl', 'rb')
# w2v_vector_30_new_w20 = open('data/w2v_vector_30_new_w20.pkl', 'rb')
w2v_vector_30_new_base64 = open('data/w2v_vector_30_new_base64.pkl', 'rb')
data0 = pickle.load(w2v_vector_30_new_30)
# data1 = pickle.load(w2v_vector_30_new_w5)
# data2 = pickle.load(w2v_vector_30_new_w10)
# data3 = pickle.load(w2v_vector_30_new_w20)
data4 = pickle.load(w2v_vector_30_new_base64)
h5f = h5py.File('data/per_all_250000.h5', 'r')
charset1 = h5f['charset'][:]
print(charset1)
charset = []
for i in charset1:
    charset.append(i.decode())
# print(data0)
# print(data1)
# print(data2)
#s = []
#for key, value in data0.items():
#    s.append(key)
#    print(np.linalg.norm(value))
#print(s, len(s))
benhuan = ['c1ccccc1', 'c1ccncc1', 'c1ncncn1']
benhuan_base64 = ['YzFjY2NjYzE=', 'YzFjY25jYzE=', 'YzFuY25jbjE=']

def vec_sum(a, b):
    for i in range(len(a)):
        a[i] = a[i] + b[i]
    return a

def vector(smiles, charset):
    smiles_vector = [0] * len(charset)
    for c in smiles:
        charset_vector = [0] * len(charset)
        for index, value in enumerate(charset):
            if c == value:
                charset_vector[index] = 1
        smiles_vector = vec_sum(smiles_vector, charset_vector)
    return smiles_vector

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

x = [0]*30
y = [0]*30
z = [0]*30
a = [0]*30
b = [0]*30
c = [0]*30
for j in range(8):
    x = vec_sum(x, data0[benhuan[0][j]])
    y = vec_sum(y, data0[benhuan[1][j]])
    z = vec_sum(z, data0[benhuan[2][j]])
for k in range(12):
    a = vec_sum(a, data4[benhuan_base64[0][k]])
    b = vec_sum(b, data4[benhuan_base64[1][k]])
    c = vec_sum(c, data4[benhuan_base64[2][k]])



d = vector(benhuan[0], charset)
e = vector(benhuan[1], charset)
f = vector(benhuan[2], charset)

# pca = PCA(n_components=30)
# d = pca.fit_transform(d)
# e = pca.fit_transform(e)
# f = pca.fit_transform(f)

p = [x, y, z]
z = [a, b, c]
q = [d, e, f]
print(p)
# print(z)
tsne = TSNE(n_components=3,
            random_state=0,
            verbose=4)
p = tsne.fit_transform(p)
z = tsne.fit_transform(z)
q = tsne.fit_transform(q)

# pca = PCA(n_components=5)
# x = pca.fit_transform(x)
# y = pca.fit_transform(y)
# z = pca.fit_transform(z)
# p = pca.fit_transform(p)
# z = pca.fit_transform(z)

p = normalization(p)
z = normalization(z)
q = normalization(q)
print(p)
print(z)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.scatter(p[:, 0], p[:, 1], marker='.', color='r', label=' ', s=1000)
ax.scatter(z[:, 0], z[:, 1], marker='.', color='y', label=' ', s=1000)
ax.scatter(q[:, 0], q[:, 1], marker='.', color='b', label=' ', s=1000)
# for i in range(0, len(charset)):
#     ax.text(x[i, 0], x[i, 1], charset[i], fontsize=15)
show()