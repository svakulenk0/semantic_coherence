# -*- coding: utf-8 -*-
'''
svakulenko
25 Mar 2018

Load word embeddings: glove and word2vec
'''
# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Load and split the dataset to train the classification model
'''
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.preprocessing.sequence import pad_sequences

from model import train
from prepare_dataset import load_dataset_splits
from load_embeddings import PATH
from embeddings import word_embeddings

# training parameters:
batch_size = 128
epochs = 10
validation_split = 0.2
# specify negative sampling strategies used e.g. 'random', 'disorder', 'distribution', 'vertical', 'horizontal' (5)
negative_sampling_strategies = ['random', 'disorder', 'distribution']
# specify embeddings, e.g. GloVe, word2vec
embedding_names = ['GloVe']

# dataset params
LATEST_SAMPLE = '291848'
vocabulary_size = 21832  # words


def load_test_data(path, input_length, sample=LATEST_SAMPLE):
    x_test = np.load(path % sample)
    # adjust input length to the layer size
    x_test = pad_sequences(x_test, padding='post', maxlen=input_length)
    return x_test


def load_training_data(strategy, sample=LATEST_SAMPLE):
    positives = np.load('./%s/words/positive_X.npy' % sample)
    n_positives = positives.shape[0]
    
    negatives = np.load('./%s/words/%s_X.npy' % (sample, strategy))
    n_negatives = negatives.shape[0]

    assert n_positives == n_negatives
    print n_positives, 'positive and negative samples'
    # merge positives + negatives for training the model to separate them
    x = np.append(positives, negatives, axis=0)
    x = pad_sequences(x, padding='post')
    y = np.append(np.ones(n_positives), np.zeros(n_negatives), axis=0)

    return x, y


def train_model(strategy, sample=LATEST_SAMPLE):
    # load dataset
    x, y = load_training_data(strategy)
    # verify the dimensions
    print 'size of development set:', x.shape[0]
    assert x.shape[0] == y.shape[0]
    input_length = x.shape[1]
    print 'max input length:', input_length

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

    # load test data

    # positive examples
    x_test_positives = load_test_data('./%s/words/test/positive_X.npy', input_length)
    n_positives = x_test_positives.shape[0]
    # verify the dimensions
    print 'size of test set positive examples:', n_positives, x_test_positives.shape[1]
    y_test_positives = np.ones(n_positives)

    # negative examples
    # uniform random
    x_test_random = load_test_data('./%s/words/test/random_X.npy', input_length)
    n_negatives = x_test_random.shape[0]
    # verify the dimensions
    print 'size of test set negative uniform random examples:', n_negatives, x_test_random.shape[1]

    # sequence disorder
    x_test_disorder = load_test_data('./%s/words/test/disorder_X.npy', input_length)
    # verify the dimensions
    print 'size of test set negative sequence disorder examples:', x_test_disorder.shape[0], x_test_disorder.shape[1]

    # vocabulary distribution
    x_test_distribution = load_test_data('./%s/words/test/distribution_X.npy', input_length)
    # verify the dimensions
    print 'size of test set negative vocabulary distribution examples:', x_test_distribution.shape[0], x_test_distribution.shape[1]
    
    y_test_negatives = np.zeros(n_negatives)
    assert n_positives == n_negatives

    for embeddings_name in embedding_names:
        label = "%s_%s_%s" % (sample, strategy, embeddings_name)
        print label
        embeddings_config = word_embeddings[embeddings_name]
        embeddings_config['matrix_path'] = PATH + 'GloVe%s.npy' % sample
        model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings_config, label, batch_size, epochs)

        # evaluate the model on each of the test set groups
        
        # true positive samples
        loss, accuracy = model.evaluate(x_test_positives, y_test_positives, verbose=1)
        print('Accuracy on true positive: %f' % (accuracy * 100))
        
        # negative samples

        loss, accuracy = model.evaluate(x_test_random, y_test_negatives, verbose=1)
        print('Accuracy on uniform random: %f' % (accuracy * 100))

        loss, accuracy = model.evaluate(x_test_disorder, y_test_negatives, verbose=1)
        print('Accuracy on sequence disorder: %f' % (accuracy * 100))

        loss, accuracy = model.evaluate(x_test_distribution, y_test_negatives, verbose=1)
        print('Accuracy on vocabulary distribution: %f' % (accuracy * 100))

        # serialize the trained model to JSON
        model_json = model.to_json()
        with open("./models/%s_model.json" % label, "w") as json_file:
            json_file.write(model_json)


if __name__ == '__main__':
    for strategy in negative_sampling_strategies:
        train_model(strategy)
