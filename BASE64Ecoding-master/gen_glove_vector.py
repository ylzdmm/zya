import numpy as np
import pickle


f = open(r'data/vectors_35_w2.txt')
lines = f.readlines()
glove_vector = {}
for line in lines:
    temp = []
    a = line.split()
    for i in a[1:]:
        temp.append(float(i))
    print(np.linalg.norm(temp))
    print(a[0])
    if a[0] == '<unk>':
        print(000)
        continue
    if a[0] == 'A':
        print(111)
        glove_vector[' '] = temp
    else:
        glove_vector[a[0]] = temp
print(len(glove_vector))
print(glove_vector[' '])
output_smiles = open('data/glove_vector_35_w2_new.pkl', 'wb')
pickle.dump(glove_vector, output_smiles)
output_smiles.close()
