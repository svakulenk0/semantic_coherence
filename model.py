# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Basic LSTM architecture in Keras with pre-trained embeddings in the input layer
for classification of dialogues given DBpedia concept annotations as input:
    1 - real (coherent) dialogue
    0 - fake (incoherent) dialogue
    generated by corrupting (preturbing) training samples

https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
'''
import gensim

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from preprocess import populate_emb_matrix_from_file

PATH_EMBEDDINGS = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2015-10/noTypes/db2vec_sg_200_5_25_5'

# define the model params
# vocab_size = len(vocabulary)


def load_embeddings(vocabulary, emb_path=PATH_EMBEDDINGS):
    # load pre-trained KG entity embeddings
    vector_model = gensim.models.Word2Vec.load(emb_path)
    print('Loaded %s entity vectors.' % len(vector_model.wv))

    # create a weight matrix for entities in training docs
    embedding_matrix = zeros((vocab_size, embeddings_dim))
    for entity, i in vocabulary.items():
        embedding_vector = vector_model.wv.get(entity)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    del vector_model
    # save embedding_matrix for entities in the training dataset
    return embedding_matrix


def train(X_train, y_train, X_text, y_test, vocabulary, input_length):
    embedding_matrix = populate_emb_matrix_from_file(vocabulary)
    embeddings_dim = 200

    # define the model architecture
    # simple
    model = Sequential()
    model.add(Embedding(len(vocabulary)+1, embeddings_dim, weights=[embedding_matrix],
                        input_length=input_length, trainable=False))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())

    # begin training validation_split=0.2, 
    model.fit(X_train, y_train, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(X_text, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
