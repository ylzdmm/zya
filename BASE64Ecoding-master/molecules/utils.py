import base64
import gzip
import pandas
import h5py
import numpy as np
import os

from jieba import xrange

base64_dictionary = {
                  'A': 64, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                  'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                  'X': 23, 'Y': 24, 'Z': 25,
                  'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36,
                  'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47,
                  'w': 48, 'x': 49, 'y': 50, 'z': 51,
                  '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61, '+': 62,
                  '/': 63, '=': 0}
def one_hot_array(i, n):
    return map(int, [ix == i for ix in xrange(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_dataset(filename, split = True):
    print(filename)
    print(os.path.isfile(filename))
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)

# def basevector(smile):
#     num = []
#     compressed = base64.b64encode(smile.encode())
#     compressed = compressed.ljust(150)
#     for i in compressed:
#         i = i/250
#         num.append(i)
#     return num
def basevector(smile):
    num = []
    smile = smile.ljust(60)
    compressed = base64.b64encode(smile.encode())
    for c in compressed:
        i = base64_dictionary[chr(c)]/64
        num.append(i)
    return num
# import datetime
# import os
# import h5py
# import numpy as np
#
# # f = h5py.File('path/filename.h5','r') #打开h5文件
# f = h5py.File('E:/tp/smiles_50k.h5', 'r')
#
# for group in f.keys():
#     print(group)
#     print("---")
#     # 根据一级组名获得其下面的组
#     group_read = f[group]
#     # 遍历该一级组下面的子组
#     for subgroup in group_read.keys():
#         print(subgroup)
#
# # print([key for key in f.keys()])
# # print(f['table'][:])