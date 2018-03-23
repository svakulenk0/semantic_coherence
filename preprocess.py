# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Preprocess input data
'''
import numpy as np

import gensim
from keras.preprocessing.sequence import pad_sequences

from process_ubuntu_dialogues import load_vocabulary, create_vocabulary
from process_ubuntu_dialogues import load_annotated_dialogues, VOCAB_ENTITIES_PATH
from process_ubuntu_dialogues import load_dialogues_words, VOCAB_WORDS_PATH

X_path = './sample127932/ubuntu127932_X.npy'
y_path = './sample127932/ubuntu127932_y.npy'

# embeddings params
embeddings = {
                'DBpedia_GlobalVectors_9_pageRank': {'matrix_path': 'embedding_matrix_DBpedia_GloVe_9PR.npy', 'dims' : 200,
                'all_path': './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'},
                
                
                'rdf2vec': {'matrix_path': 'embedding_matrix_rdf2vec.npy', 'dims' : 200,
                'all_path': '/home/cochez/biasedRDF2Vec/PageRank/db2vec_sg_200_5_25_5'},                

                'word2vec': {'matrix_path': 'embedding_matrix_word2vec.npy', 'dims' : 300,
                'all_path': './embeddings/GoogleNews-vectors-negative300.bin'},
                
                'GloVe': {'matrix_path': 'embedding_matrix_GloVe.npy', 'dims' : 300,
                'all_path': './embeddings/glove.840B.300d.txt'}
             }

# entity label of the format: <http://dbpedia.org/resource/Albedo>
# DBPEDIA_GLOBAL_PR = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'
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


def prepare_dataset(encode_dialogue=load_dialogues_words, vocab_path=VOCAB_WORDS_PATH, n_dialogues=None):
    '''
    encode_dialogue is a function: load_annotated_dialogues for the dialogue
    as a sequence of entities for entity embeddings representation
    or load_dialogues_words for the dialogue as a sequence of words
    for word embeddings representation
    '''
    # create_vocabulary()

    vocabulary = load_vocabulary(vocab_path)

    # load correct and incorrect examples
    # dialogue as a sequence of entities for entity embeddings
        # X, labels = load_annotated_dialogues(vocabulary, n_dialogues)
    # dialogue as a sequence of words for word embeddings
    X, labels = encode_dialogue(vocabulary, n_dialogues)
    print X
    print X.shape[0], 'dialogues', X.shape[1], 'max entities/words per dialogue'
    print labels
    # save dataset
    # save embedding_matrix for entities in the training dataset
    np.save(X_path, X)
    np.save(y_path, labels)


def load_text_gloves(embeddings=embeddings['GloVe']):
    vocabulary = load_vocabulary()
    # strip vocabulary to surface form

    # create a weight matrix for entities in training docs

    # embedding_matrix = np.zeros((len(vocabulary)+1, embeddings['dims']))
    # with open(emb_path) as embs_file:
    #     embedding_matrix = load_embeddings(embeddings['all_path'], embedding_matrix, vocabulary)
    #     # save embedding_matrix for entities in the training dataset
    #     np.save(embeddings['matrix_path'], embedding_matrix)
    # print embedding_matrix


def populate_emb_matrix_from_file(embeddings_name, limit_n=None):
    # create_vocabulary(limit_n)
    vocabulary = load_vocabulary()
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings[embeddings_name]['dims']))
    with open(embeddings[embeddings_name]['all_path']) as embs_file:
        embedding_matrix = load_embeddings(embs_file, embedding_matrix, vocabulary)
        # save embedding_matrix for entities in the training dataset
        np.save(embeddings[embeddings_name]['matrix_path'], embedding_matrix)
    print embedding_matrix
    # return embedding_matrix


def load_embeddings_gensim(embeddings_name):
    vocabulary = load_vocabulary()
    # create a weight matrix for entities in training docs
    embedding_matrix = np.zeros((len(vocabulary)+1, embeddings[embeddings_name]['dims']))
        
    # load embeddings binary model with gensim for word2vec and rdf2vec embeddings
    #model = gensim.models.KeyedVectors.load_word2vec_format(embeddings[embeddings_name]['all_path'], binary=True)
    #embedded_entities = model.wv
    model = gensim.models.Word2Vec.load(embeddings[embeddings_name]['all_path'])
    embedded_entities = model.wv


    # test loaded model on a similarity example
    # model.most_similar(positive=['dbr:Rocky'], topn=100)  # rdf2vec
    # model.most_similar(positive=['rocky'], topn=100)  # word2vec
    
    count = 0
    for entity, entity_id in vocabulary.items():
        count += 1
        if count % 100 == 0:
            print str(count) + " done"
        #print entity, entity_id
        # strip entity label format to rdf2vec label format
        #rdf2vec_entity_label = 'dbr:%s' % entity.split('/')[-1]
        #print rdf2vec_entity_label
        rdf2vec_entity_label = '<' + entity + '>'
        if rdf2vec_entity_label in embedded_entities:
            embedding_matrix[entity_id] = embedded_entities[rdf2vec_entity_label]
        else:
            print "missing entity" + rdf2vec_entity_label

    # save embedding_matrix for entities in the training dataset
    np.save('embedding_matrix.npy', embedding_matrix)
    print embedding_matrix


def load_embeddings(embeddings, embedding_matrix, vocabulary):
    words = 0
    vocabulary_entities = vocabulary.keys()
    # embeddings in a text file one per line for Global vectors and glove word embeddings
    for line in embeddings:
        values = line.split()
        # match the entity labels in vector embeddings
        word = values[0]
        # word = word[1:-1]  # Dbpedia global vectors strip <> to match the entity labels 
        print word
        if word in vocabulary_entities:
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
    # prepare_dataset(encode_dialogue=load_annotated_dialogues, vocab_path=VOCAB_ENTITIES_PATH)
    # prepare_dataset(encode_dialogue=load_dialogues_words, vocab_path=VOCAB_WORDS_PATH)
    # embeddings_name = 'DBpedia_GlobalVectors_9_pageRank'
    # populate_emb_matrix_from_file(embeddings_name)
    embeddings_name = 'rdf2vec'
    load_embeddings_gensim(embeddings_name)
    # load_text_gloves()
    # populate_emb_matrix_from_file(embeddings['word2vec'])
