# -*- coding: utf-8 -*-
'''
cochez
11 Apr 2018

Create random embeddings
'''
import numpy as np
from prepare_dataset import load_vocabulary


def create_embeddings_random(dimension=200, voc='entities'):
    '''
    from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

    dimension <int> dimension of the random vectors
    vocabulary <str> 'entities' or 'words'
    '''
    vocabulary = load_vocabulary('../data/%s/vocab.pkl' % voc)
    label = "random_vectors"
    try:
        print label
        # create a weight matrix for entities in training docs
        embedding_matrix = np.zeros((len(vocabulary), dimension))
        for vocabulary_term in vocabulary.keys():
            embedding_vector = np.random.rand(dimension)
            embedding_matrix[vocabulary[vocabulary_term]] = embedding_vector
        # save embedding_matrix for entities in the training dataset
        np.save('../data/%s/embeddings/'%voc+label+'.npy', embedding_matrix)
    except Exception as e:
        print e


if __name__ == '__main__':
    # create random entity embeddings
    create_embeddings_random()
    # create random word embeddings
    create_embeddings_random(300, 'words')
