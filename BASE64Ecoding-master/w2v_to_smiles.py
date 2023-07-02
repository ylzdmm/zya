import pickle
import numpy
from gensim.models import word2vec 
w2v_vector = open('w2v_vector_120.pkl', 'rb')                                                                                                                           
w2v_vector = pickle.load(w2v_vector)
a = [0]*120
a = numpy.array(a)
model = word2vec.Word2Vec.load('w2v_120.model')
res = model.wv.similar_by_vector(a, topn = 2)
print(res)

