# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Train CNN to classify dialogues using DBpedia entity annotations as input
'''
import numpy as np

from process_ubuntu_dialogues import load_annotated_dialogues, load_vocabulary
from model import 

# dataset params
vocabulary_size = 19659
input_length = 254
X_path = 'ubuntu127932_X.npy'
y_path = 'ubuntu127932_y.npy'

# load dataset
X = np.load(X_path)
y = np.load(labels_path)

# labels = np.full((X.shape[0]), 1)
# input_length = X.shape[1]
# print 'max input length:', input_length


# embeddings params
embeddings = {
                'DBpedia_GlobalVectors': {'9_pageRank': {'path': 'embedding_matrix_PR.npy', 'dims' : 200}},
                # 'word2vec': {'path': 'embedding_matrix_PR.npy', 'dims' : 200},
                # 'GloVe': {'path': 'embedding_matrix_PR.npy', 'dims' : 200}
             }


train(X, y, X, y, vocabulary_size, input_length, embeddings['DBpedia_GlobalVectors']['9_pageRank'])
