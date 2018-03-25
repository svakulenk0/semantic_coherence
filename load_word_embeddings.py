# -*- coding: utf-8 -*-
'''
svakulenko
25 Mar 2018

Load word embeddings: glove and word2vec
'''
import numpy as np

from embeddings import word_embeddings
from process_ubuntu_dialogues import load_vocabulary, VOCAB_WORDS_PATH
from load_embeddings import PATH


def load_embeddings(embeddings, embedding_matrix, vocabulary):
    words = 0
    # embeddings in a text file one per line for Global vectors and glove word embeddings
    for line in embeddings:
        values = line.split()
        # match the entity labels in vector embeddings
        word = values[0]
        # print word
        if word in vocabulary:
            print word
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embedding_matrix[vocabulary[word]] = embedding_vector
            words += 1
            if words >= len(vocabulary):
                return embedding_matrix
    return embedding_matrix


def load_embeddings_lines(embeddings_config, label, vocabulary):
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_config['dims']))
    with open(embeddings_config['path']) as embs_file:
        embedding_matrix = load_embeddings(embs_file, embedding_matrix, vocabulary)
        # save embedding_matrix for entities in the training dataset
        np.save(PATH+label+'.npy', embedding_matrix)
    return embedding_matrix


def load_embeddings_gensim(embeddings_config, label, vocabulary):
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_config['dims']))
        
    # load embeddings binary model with gensim for word2vec and rdf2vec embeddings
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_config['path'], binary=True)
    embedded_entities = model.wv
    
    for word, word_id in vocabulary.items():
        # strip entity label format to rdf2vec label format
        print word
        if word in embedded_entities:
            embedding_matrix[word_id] = model.wv[word]

    # save embedding_matrix for entities in the training dataset
    np.save(PATH+label+'.npy', embedding_matrix)
    return embedding_matrix


if __name__ == '__main__':

    vocabulary = load_vocabulary(VOCAB_WORDS_PATH)

    label = 'GloVe'
    print label
    embedding_matrix = load_embeddings_lines(word_embeddings[label], label, vocabulary)
    # number of non-zero rows, i.e. entities with embeddings
    print len(np.where(embedding_matrix.any(axis=1))[0])

    label = 'word2vec'
    print label
    embedding_matrix = load_embeddings_gensim(word_embeddings[label], label, vocabulary)
    # number of non-zero rows, i.e. entities with embeddings
    print len(np.where(embedding_matrix.any(axis=1))[0])
