# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Load embeddings
'''
import numpy as np

import gensim
from keras.preprocessing.sequence import pad_sequences

from process_ubuntu_dialogues import load_vocabulary, create_vocabulary
from process_ubuntu_dialogues import load_annotated_dialogues, VOCAB_ENTITIES_PATH
from process_ubuntu_dialogues import load_dialogues_words, VOCAB_WORDS_PATH
from embeddings import *

PATH = './embeddings_npy/'

def load_embeddings(embeddings, embedding_matrix, vocabulary):
    words = 0
    vocabulary_entities = vocabulary.keys()
    # embeddings in a text file one per line for Global vectors and glove word embeddings
    for line in embeddings:
        values = line.split()
        # match the entity labels in vector embeddings
        word = values[0]
        word = word[1:-1]  # Dbpedia global vectors strip <> to match the entity labels 
        print word
        if word in vocabulary_entities:
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embedding_matrix[vocabulary[word]] = embedding_vector
            
            words += 1
            if words >= len(vocabulary):
                return embedding_matrix
    return embedding_matrix


def load_embeddings_lines(embeddings_config, label):
    vocabulary = load_vocabulary()
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_config['dims']))
    with open(embeddings_config['path']) as embs_file:
        embedding_matrix = load_embeddings(embs_file, embedding_matrix, vocabulary)
        # save embedding_matrix for entities in the training dataset
        np.save(PATH+label+'.npy', embedding_matrix)
    return embedding_matrix


def load_embeddings_gensim(embeddings_config, label):
    vocabulary = load_vocabulary()
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_config['dims']))
        
    # load embeddings binary model with gensim for word2vec and rdf2vec embeddings
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_config['path'], binary=True)
    embedded_entities = model.wv
    
    for entity, entity_id in vocabulary.items():
        # strip entity label format to rdf2vec label format
        rdf2vec_entity_label = 'dbr:%s' % entity.split('/')[-1]
        print rdf2vec_entity_label
        if rdf2vec_entity_label in embedded_entities:
            embedding_matrix[entity_id] = model.wv[entity]

    # save embedding_matrix for entities in the training dataset
    np.save(PATH+label+'.npy', embedding_matrix)
    return embedding_matrix


if __name__ == '__main__':

    for embeddings_name, config in embeddings['GlobalVectors'].items():
        label = 'GlobalVectors_' + embeddings_name
        print label
        load_embeddings_lines(config, label)

    for embeddings_name, config in embeddings['rdf2vec'].items():
        label = 'rdf2vec_' + embeddings_name
        print label
        load_embeddings_gensim(config, label)
