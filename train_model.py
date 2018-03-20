# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Train CNN to classify dialogues using DBpedia entity annotations as input
'''
import numpy as np

from model import train
from preprocess import X_path, y_path, embeddings


# dataset params
vocabulary_size = 19659
input_length = 254

# load dataset
X = np.load(X_path)
print X
y = np.load(y_path)
print y

# labels = np.full((X.shape[0]), 1)
# input_length = X.shape[1]
# print 'max input length:', input_length

train(X, y, X, y, vocabulary_size, input_length, embeddings['DBpedia_GlobalVectors']['9_pageRank'])
# train(X, y, X, y, vocabulary_size, input_length, embeddings['GloVe'])
