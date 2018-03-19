# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Preprocess input data
'''
import numpy as np

import gensim
from keras.preprocessing.sequence import pad_sequences

from process_ubuntu_dialogues import load_vocabulary, create_vocabulary, load_annotated_dialogues

X_path = 'ubuntu127932_X.npy'
y_path = 'ubuntu127932_y.npy'

# entity label of the format: <http://dbpedia.org/resource/Albedo>
DBPEDIA_GLOBAL_PR = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'
# RDF2VEC = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2015-10/noTypes/db2vec_sg_200_5_25_5'
# go over the dataset and create vocabulary of concepts mentioned in the dataset, save it

# load the vocabulary word2id

def preprocess(docs, vocabulary, max_length):
    # process input documents
    # encode replace concept/resource names with int ids from vocabulary
    encoded_docs = [[vocabulary[e] for e in d] for d in docs ]
    # print encoded_docs
    # pad documents to a max number of concepts per document
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs


def prepare_dataset(n_dialogues=10):
    # create_vocabulary()

    vocabulary = load_vocabulary()

    # load correct and incorrect examples
    X, labels = load_annotated_dialogues(vocabulary, n_dialogues)
    print X
    print labels
    # save dataset
    # save embedding_matrix for entities in the training dataset
    np.save(X_path, X)
    np.save(y_path, labels)


def populate_emb_matrix_from_file(limit_n=None, embeddings_dim=200, emb_path=DBPEDIA_GLOBAL_PR):
    # create_vocabulary(limit_n)
    vocabulary = load_vocabulary()
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_dim))
    with open(emb_path) as embs_file:
        embedding_matrix = load_embeddings(embs_file, embedding_matrix, vocabulary)
        # save embedding_matrix for entities in the training dataset
        np.save('embedding_matrix.npy', embedding_matrix)
    print embedding_matrix
    # return embedding_matrix


def load_embeddings_gensim(embeddings_dim=200):
    vocabulary = load_vocabulary()
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings_dim))
        
    # load embeddings binary model with gensim for word2vec and rdf2vec embeddings
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    embedded_entities = model.wv.keys()
    # test loaded model on a similarity example
    # model.most_similar(positive=['dbr:Rocky'], topn=100)  # rdf2vec
    # model.most_similar(positive=['rocky'], topn=100)  # word2vec
    
    for entity, entity_id in vocabulary.items():
        if entity in embedded_entities:
            embedding_matrix[entity_id] = model.wv[entity]

    # save embedding_matrix for entities in the training dataset
    np.save('embedding_matrix.npy', embedding_matrix)
    print embedding_matrix


def load_embeddings(embeddings, embedding_matrix, vocabulary):
    words = 0
    # embeddings in a text file one per line for Global vectors and glove word embeddings
    for line in embeddings:
        values = line.split()
        # strip <> to match the entity labels in global vectors 
        word = values[0][1:-1]
        # print word
        if word in vocabulary.keys():
            embedding_vector = np.asarray(values[1:], dtype='float32')
            # print word
            # print embedding_vector
            # return
            embedding_matrix[vocabulary[word]] = embedding_vector
            
            words += 1
            if words >= len(vocabulary):
                return embedding_matrix

    return embedding_matrix


if __name__ == '__main__':
    # encode the whole datase and save it into 2 matrices X, y
    prepare_dataset()
    # populate_emb_matrix_from_file()
