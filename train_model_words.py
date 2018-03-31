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

from model import train
from prepare_dataset import load_dataset_splits
from load_embeddings import PATH
from embeddings import word_embeddings

# training parameters:
batch_size = 128
epochs = 5
validation_split = 0.2
# specify negative sampling strategies used e.g. 'random', 'vertical', 'horizontal'
negative_sampling_strategies = ['random']
# specify embeddings, e.g. GloVe, word2vec
embedding_names = ['GloVe']

# dataset params
sample = '291848'
vocabulary_size = 21832  # unique words


for strategy in negative_sampling_strategies:
    # load dataset
    positives = np.load('./%s/words/positive_X.npy' % sample)
    n_positives = positives.shape[0]
    
    negatives = np.load('./%s/words/%s_X.npy' % (sample, strategy))
    n_negatives = negatives.shape[0]

    assert n_positives == n_positives
    print n_positives, 'positive and negative samples'
    # merge positives + negatives for training the model to separate them
    x = np.append(positives, negatives, axis=0)
    y = np.append(np.ones(n_positives), np.zeros(n_negatives), axis=0)

    # verify the dimensions
    print 'size of development set:', x.shape[0]
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

    for embeddings_name in embedding_names:
        label = "%s_%s" % (strategy, embeddings_name)
        print label
        embeddings_config = word_embeddings[embeddings_name]
        embeddings_config['matrix_path'] = PATH + embeddings_name + sample + '.npy'
        model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings_config, label, batch_size, epochs)

        # evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print('Accuracy: %f' % (accuracy * 100))

        # serialize the trained model to JSON
        model_json = model.to_json()
        with open("./models/%s_%s_model.json" % (sample, label), "w") as json_file:
            json_file.write(model_json)
    
    # # serialize weights to HDF5
    # model.save_weights('./models/%s_weights_172098.h5' % label)
    # print("Saved model to disk")
