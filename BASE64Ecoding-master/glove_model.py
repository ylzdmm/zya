from __future__ import print_function
import argparse
import pprint
import gensim
import pickle
import pandas as pd
import h5py
import numpy as np
from glove import Glove
from glove import Corpus
from sklearn import preprocessing

data = pd.read_hdf('/data/tp/data/zinc-1.h5', 'table')
smiles = data['smiles']
sentences = []
for line in smiles:
    sts = [x for x in line]
    sts.extend([' '])
    sentences.append(sts)


corpus_model = Corpus()
corpus_model.fit(sentences, window=30)
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)


glove = Glove(no_components=35, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
glove.save('model/glove.model')
glove = Glove.load('model/glove.model')
corpus_model.save('model/corpus.model')
corpus_model = Corpus.load('model/corpus.model')
print(len(glove.word_vectors))
print(glove.word_vectors[glove.dictionary['C']])
print(np.linalg.norm(glove.word_vectors[glove.dictionary[' ']]))
print(glove.most_similar('C', number=10))
    
#filename = '/data/tp/data/per_all_250000.h5'
#h5f = h5py.File(filename, 'r')
#charset1 = h5f['charset'][:]
#charset = []
#for i in charset1:
#    charset.append(i.decode())

#vector1 = []
#glove_vector = {}
#for s in charset:
#    print(s)
#    vector1.append(glove.word_vectors[glove.dictionary[s]])
#vector = preprocessing.normalize(vector1, norm='l2')
#for i in range(len(charset)):
#    print(charset[i])
#    glove_vector[charset[i]] = vector[i]
#    print(np.linalg.norm(vector[i]))
#output_smiles = open('glove_vector_30_lr(0.01).pkl', 'wb')
#pickle.dump(glove_vector, output_smiles)
#output_smiles.close()



