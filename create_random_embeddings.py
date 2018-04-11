# -*- coding: utf-8 -*-
'''
cochez
11 Apr 2018

Create random embeddings
'''
import numpy as np
from prepare_dataset import load_vocabulary, LATEST_SAMPLE

PATH = './embeddings_npy/'

def create_embeddings_random(embeddings_config, label, vocabulary):
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    dimension = embeddings_config['dims']
    embedding_matrix = np.zeros((len(vocabulary)+1, dimension))
    for vocabulary_term in vocabulary.keys():
        embedding_vector = np.random.rand(dimension)
        embedding_matrix[vocabulary[vocabulary_term]] = embedding_vector
    # save embedding_matrix for entities in the training dataset
    np.save(PATH+label+'.npy', embedding_matrix)

def load_random_embeddings(sample=LATEST_SAMPLE):
    # dataset params
    entity_vocabulary = load_vocabulary('./%s/entities/vocab.pkl' % sample)

    embeddings_name = "random_vectors"
    config = {'dims' : 200, 'path': ''}
    label = 'GlobalVectors_' + embeddings_name
    try:
        print label
        create_embeddings_random(config, label, entity_vocabulary)
    except Exception as e:
            print e

    

if __name__ == '__main__':
    # load_glove_word_embeddings()
    # path to the data: './291848/entities/vocab.pkl'
    load_random_embeddings()
