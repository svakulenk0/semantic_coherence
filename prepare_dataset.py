# -*- coding: utf-8 -*-
'''
svakulenko
23 Mar 2018

Preprocess input data
'''
import numpy as np
import json
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences

from annotate_ubuntu_dataset import ANNOTATION_FILE

LATEST_SAMPLE = '291848'
DEV_DATA_PATH = 'development_set.jl'
TEST_DATA_PATH = 'test_set.jl'
VOCAB_ENTITIES_PATH = './%s/entities/vocab.pkl'
VOCAB_WORDS_PATH = './%s/words/vocab.pkl'

POSITIVE_ENTITIES_DEV = './%s/entities/positive_X.npy'
POSITIVE_WORDS_DEV = './%s/words/positive_X.npy'

POSITIVE_ENTITIES_TEST = './%s/entities/test/positive_X.npy'
POSITIVE_WORDS_TEST = './%s/words/test/positive_X.npy'


def load_vocabulary(path):
    with open(path, 'rb') as f:
        vocabulary = pickle.load(f)
        print 'Loaded vocabulary with', len(vocabulary), 'keys'
        return vocabulary


def encode_data_set(sample=LATEST_SAMPLE, source=DEV_DATA_PATH, targets=(POSITIVE_ENTITIES_DEV, POSITIVE_WORDS_DEV)):
    entity_vocabulary = load_vocabulary(VOCAB_ENTITIES_PATH % sample)
    word_vocabulary = load_vocabulary(VOCAB_WORDS_PATH % sample)

    encoded_docs_entities = []
    encoded_docs_words = []

    with open(source, "rb") as entities_file:
        for line in entities_file:
            annotation = json.loads(line)
            encoded_docs_entities.append([entity_vocabulary[entity] for entity in annotation['entity_URIs']])
            encoded_docs_words.append([word_vocabulary[word] for entity in annotation['surface_forms']
                                                             for word in entity.split()])
    assert len(encoded_docs_entities) == len(encoded_docs_words)
    
    np.save(targets[0] % sample, encoded_docs_entities)
    np.save(targets[1] % sample, encoded_docs_words)

    print len(encoded_docs_entities), 'documents encoded'


def measure_vocabulary_distribution(path=ANNOTATION_FILE, sample=LATEST_SAMPLE):
    # load vocabularies
    entity_vocabulary = load_vocabulary(VOCAB_ENTITIES_PATH % sample)
    word_vocabulary = load_vocabulary(VOCAB_WORDS_PATH % sample)
    # produce vocabulary distributions
    # entity distribution across dialogues
    entity_distribution = Counter()
    # entity word distribution across dialogues: entity id -> count
    word_distribution = Counter()

    with open(path, "rb") as entities_file:
        for line in entities_file:
            annotation = json.loads(line)
            for entity in annotation['entity_URIs']:
                entity_distribution[entity_vocabulary[entity]] += 1
            for entity in annotation['surface_forms']:
                for word in entity.split():
                    word_distribution[word_vocabulary[word]] += 1

    print entity_distribution
    print word_distribution


def create_vocabularies(path=ANNOTATION_FILE, sample=LATEST_SAMPLE):
    entity_vocabulary = {}
    word_vocabulary = {}

    with open(path, "rb") as entities_file:
        for line in entities_file:
            annotation = json.loads(line)
            for entity in annotation['entity_URIs']:
                if entity not in entity_vocabulary:
                    entity_vocabulary[entity] = len(entity_vocabulary)
            for entity in annotation['surface_forms']:
                for word in entity.split():
                    if word not in word_vocabulary:
                        word_vocabulary[word] = len(word_vocabulary)
    # save vocabularies
    with open(VOCAB_ENTITIES_PATH % sample, 'wb') as f:
        pickle.dump(entity_vocabulary, f)
    print 'Saved vocabulary with', len(entity_vocabulary.keys()), 'entities'

    with open(VOCAB_WORDS_PATH % sample, 'wb') as f:
        pickle.dump(word_vocabulary, f)
    print 'Saved vocabulary with', len(word_vocabulary.keys()), 'words'


def separate_test_set(path=ANNOTATION_FILE, test_set_size=5000):
    import random

    with open(path, "rb") as f:
        data = f.readlines()
    # shuffle
    random.shuffle(data)
    # split
    test_data = data[:test_set_size]
    development_data = data[test_set_size:]
    # write
    with open('test_set.jl', "wb") as test_file, open(DEV_DATA_PATH, "wb") as development_file:
        test_file.writelines(test_data)
        development_file.writelines(development_data)


def preprocess(docs, vocabulary, max_length):
    '''
    process input documents
    '''
    # encode replace concept/resource names with int ids from vocabulary
    encoded_docs = [[vocabulary[e] for e in d] for d in docs ]
    # pad documents to a max number of concepts per document
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs


# def prepare_dataset(encode_dialogue=load_dialogues_words, vocab_path=VOCAB_WORDS_PATH, n_dialogues=None):
#     '''
#     encode_dialogue is a function: load_annotated_dialogues for the dialogue
#     as a sequence of entities for entity embeddings representation
#     or load_dialogues_words for the dialogue as a sequence of words
#     for word embeddings representation
#     '''
#     # create_vocabulary()

#     vocabulary = load_vocabulary(vocab_path)

#     # load correct and incorrect examples
#     # dialogue as a sequence of entities for entity embeddings
#         # X, labels = load_annotated_dialogues(vocabulary, n_dialogues)
#     # dialogue as a sequence of words for word embeddings
#     X, labels = encode_dialogue(vocabulary, n_dialogues)
#     print X
#     print X.shape[0], 'dialogues', X.shape[1], 'max entities/words per dialogue'
#     print labels
#     # save dataset
#     # save embedding_matrix for entities in the training dataset
#     np.save(X_path, X)
#     np.save(y_path, labels)


def load_dataset_splits(X_path, y_path, test_split=0.2, validation_split=0.2):
    # load dataset
    data = np.load(X_path)
    labels = np.load(y_path)

    input_length = data.shape[1]
    print 'max input length:', input_length


    # split the data into a test set and a training set
    # https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(test_split * data.shape[0])

    x = data[:-num_validation_samples]
    y = labels[:-num_validation_samples]
    x_test = data[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]

    # split the training set into a training set and a validation set
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    num_validation_samples = int(validation_split * x.shape[0])

    x_train = x[:-num_validation_samples]
    y_train = y[:-num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]

    return x_train, y_train, x_val, y_val, x_test, y_test, input_length


if __name__ == '__main__':
    # create_vocabularies()
    # encode_data_set(source=DEV_DATA_PATH, targets=(POSITIVE_ENTITIES_DEV, POSITIVE_WORDS_DEV))
    # encode_data_set(source=TEST_DATA_PATH, targets=(POSITIVE_ENTITIES_TEST, POSITIVE_WORDS_TEST))
    measure_vocabulary_distribution()
