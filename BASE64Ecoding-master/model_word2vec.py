import numpy as np
from keras.layers import Input, Embedding,  Lambda
from keras.models import Model, load_model
from keras.utils import plot_model
import keras.backend as K
import pandas as pd
import os
import h5py
import pickle
import base64

def getdata(smiles):
    sentences = []
    data = []
    for line in smiles:
        data.append(line.strip())
        sts = [x for x in line]
        sentences.append(sts)
    return data,sentences

def bulid_dic(sentences):
    charset = []
    words = {}
    nb_sentence = 0
    total = 0.
    for d in sentences:
        d.extend([' '])
        nb_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
                charset.append(w)
            words[w] += 1
            total += 1
    print(len(words))
    words = {i:j for i,j in words.items() if j >= min_count}
    id2word = {i+1:j for i,j in enumerate(words)}
    word2id = {j:i for i,j in id2word.items()}
    nb_word = len(words)+1
    subsamples = {i:j/total for i,j in words.items() if j/total > subsample_t}
    subsamples = {i:subsample_t/j+(subsample_t/j)**0.5 for i,j in subsamples.items()}
    subsamples = {word2id[i]:j for i,j in subsamples.items() if j < 1.}
    return nb_sentence,id2word,word2id,nb_word,subsamples,charset

def data_generator(word2id,subsamples,data):
    x, y = [], []
    count = 0
    for d in data:
        d = [0]*window + [word2id[w] for w in d if w in word2id] + [0]*window
        r = np.random.random(len(d))
        for i in range(window, len(d)-window):
            if d[i] in subsamples and r[i] > subsamples[d[i]]:
                continue
            x.append(d[i-window:i]+d[i+1:i+1+window])
            y.append([d[i]])
        count += 1
        # if count == nb_sentence_per_batch:
    x, y = np.array(x), np.array(y)
    z = np.zeros((len(x), 1))
    return [x, y], z

def build_w2vm(word_size,window,nb_word,nb_negative):
    K.clear_session()
    input_words = Input(shape=(window*2,), dtype='int32')
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)
    
    target_word = Input(shape=(1,), dtype='int32')
    negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
    samples = Lambda(lambda x: K.concatenate(x))([target_word, negatives])
    
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    softmax = Lambda(lambda x: 
                        K.softmax((K.batch_dot(x[0], K.expand_dims(x[1],2))+x[2])[:,:,0])
                    )([softmax_weights,input_vecs_sum,softmax_biases])
    model = Model(inputs=[input_words, target_word], outputs=softmax)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def most_similar(word2id,w): 
    model = load_model(modelname) 
    embeddings = model.get_weights()[0]
    #print(np.linalg.norm(embeddings[0]))
    normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5
    #print(len(normalized_embeddings),len(normalized_embeddings[0]), normalized_embeddings[0])
    v = normalized_embeddings[word2id[w]]
    # v = embeddings[word2id[w]]
    #sims = np.dot(normalized_embeddings, v)
    #sort = sims.argsort()[::-1]
    #sort = sort[sort > 0]
    #return [(id2word[i],sims[i]) for i in sort[:10]]
    return v

if __name__ == '__main__': 
    data = pd.read_hdf('/data/tp/data/zinc-1.h5', 'table')
    smiles = data['smiles']
    word_size = 35
    window = 2
    nb_negative = 1
    min_count = 0
    nb_worker = 4
    nb_epoch = 10
    subsample_t = 1e-5
    nb_sentence_per_batch = 249455
    modelname = 'model/word2vec/nb_negative_1/word2vec_w2_35.h5'
    data, sentences = getdata(smiles)
    nb_sentence, id2word, word2id, nb_word, subsamples, charset = bulid_dic(sentences)
    print(nb_sentence)
    print(charset)
    ipt, opt = data_generator(word2id, subsamples, data)
    print(len(ipt[0]))
    #model = build_w2vm(word_size, window, nb_word, nb_negative)
    #model.fit(ipt, opt,
              # steps_per_epoch=int(nb_sentence/nb_sentence_per_batch),
    #          epochs=nb_epoch,
    #          batch_size=128
    #          )
    #model.save(modelname)

    filename = '/data/tp/data/per_all_250000.h5'
    h5f = h5py.File(filename, 'r')
    charset = h5f['charset'][:]

    print(charset)
    w2v_vector = {}
    for s in charset:
        s = s.decode()
        print(s)
        w2v_vector[s] = most_similar(word2id, s)
    print(w2v_vector[' '])
    output_smiles = open('data/w2v_vector_35_new_w2_n1.pkl', 'wb')
    pickle.dump(w2v_vector, output_smiles)
    output_smiles.close()

    output_id2word = open('data/id2word_w2_35_n1.pkl', 'wb')
    output_word2id = open('data/word2id_w2_35_n1.pkl', 'wb')
    pickle.dump(id2word, output_id2word)
    pickle.dump(word2id, output_word2id)

    output_id2word.close()
    output_word2id.close()

