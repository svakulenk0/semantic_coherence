# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Preprocess input data
'''
from numpy import zeros
from numpy import asarray

from keras.preprocessing.sequence import pad_sequences

from process_ubuntu_dialogues import load_vocabulary, create_vocabulary

# entity label of the format: <http://dbpedia.org/resource/Albedo>
DBPEDIA_GLOBAL_PR = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'

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


def populate_emb_matrix_from_file(limit_n=None, embeddings_dim=200, emb_path=DBPEDIA_GLOBAL_PR):
    create_vocabulary(limit_n)
    vocabulary = load_vocabulary()
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    embedding_matrix = zeros((len(vocabulary)+1, embeddings_dim))
    with open(emb_path) as embs_file:
        embedding_matrix = load_embeddings(embs_file, embedding_matrix, vocabulary)
        # TODO save embedding_matrix for entities in the training dataset
        np.save('embeddings_layer.npy', embedding_matrix)
    print embedding_matrix
    # return embedding_matrix


def load_embeddings(embeddings, embedding_matrix, vocabulary):
    words = 0
    for line in embeddings:
        values = line.split()
        # strip <> to match the entity labels in global vectors 
        word = values[0][1:-1]
        # print word
        if word in vocabulary.keys():
            embedding_vector = asarray(values[1:], dtype='float32')
            # print word
            # print embedding_vector
            # return
            embedding_matrix[vocabulary[word]] = embedding_vector
            
            words += 1
            if words >= len(vocabulary):
                return embedding_matrix

    return embedding_matrix


if __name__ == '__main__':
    populate_emb_matrix_from_file()
