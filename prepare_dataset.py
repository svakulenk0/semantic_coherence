# -*- coding: utf-8 -*-
'''
svakulenko
23 Mar 2018

Preprocess input data
'''
import numpy as np

# input matrix with random negative sampling
X_path = './sample127932/ubuntu127932_X.npy'
y_path = './sample127932/ubuntu127932_y.npy'


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


def load_dataset_splits(test_split=0.2, validation_split=0.2):
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
