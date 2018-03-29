# -*- coding: utf-8 -*-
'''
svakulenko
29 Mar 2018

Run pre-trained model (inference) on different types of data samples
# 1) True positive samples: all real dialogues from the Ubuntu dataset
# 2) True negative samples: drawn uniformly at random from the vocabulary
# 3) True negative samples: drawn from the vocabulary frequency (count) distribution
# 4) True negative samples: 2 dialogues mixed by horizontal split
# 5) True negative samples: 2 dialogues mixed by vertical split
'''
import numpy as np
import keras


class Tester():

    def __init__(self, sample, embeddings_name, limit=10):
        '''
        Load model on init
        limit <int> predict only the first n samples in each dataset
        '''
        model_path = './models/%s/%s_model.json' % (sample, embeddings_name)
        weights_path = './models/%s/%s.h5' % (sample, embeddings_name)
        vocabulary_path = './%s/vocab_words.pkl' % sample
        self.model = self.load_model(model_path, weights_path)
        self.limit = limit

    def load_model(self, model_path, weights_path):
        # 1. Load model pre-trained model
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        # print loaded_model_json
        model = keras.models.model_from_json(loaded_model_json)
        # load pre-trained model weights
        model.load_weights(weights_path)
        print('Model loaded.')
        return model

    def load_data(self, X_path):
        '''
        # 2. Load word data samples and split positive/negative pairs
        '''
        X = np.load(X_path)
        print 'input shape:', X.shape
        positive = X[0:][::2] # even
        negative = X[1:][::2] # odd
        return positive, negative

    def test_model(self, X):
        '''
        3. Run model (inference)

        limit <int> predict only the first n samples in each dataset
        '''
        return self.model.predict(X[:self.limit])


if __name__ == '__main__':
    # The best word model obtained by training on negative samples drawn uniformly at random (acc: 98.4 on its test set): sample172098
    # sample172098_new
    sample = 'sample172098'
    embeddings_name = 'GloVe'
    # predict only the first n samples in each dataset
    limit = 2
    # initialize a Tester object to the model
    tester = Tester(sample, embeddings_name, limit)

    # test on:
    # 1) True positive samples: all real dialogues from the Ubuntu dataset
    strategy = 'random'
    X_path = './%s/words_%s_X.npy' % (sample, strategy)
    positive, negative = tester.load_data(X_path)
    positive_results = tester.test_model(positive)
    print positive_results

    # 2) True negative samples: drawn uniformly at random from the vocabulary
    random_uniform_results = tester.test_model(negative)
    print random_uniform_results

    # 3) True negative samples: drawn from the vocabulary frequency (count) distribution
    sample2 = 'sample172098_new'
    strategy = 'random'
    X_path = './%s/words_%s_X.npy' % (sample2, strategy)
    positive, negative = tester.load_data(X_path)
    # pad with 0s to fit the input length of the model
    negative = np.pad(negative, (0, 5), 'constant')
    random_voc_distr_results = tester.test_model(negative)
    print random_voc_distr_results

    # 4) True negative samples: 2 dialogues mixed by horizontal split
    strategy = 'horizontal'
    X_path = './%s/words_%s_X.npy' % (sample, strategy)
    positive, negative = tester.load_data(X_path)
    horizontal_split_results = tester.test_model(negative)
    print horizontal_split_results

    # 5) True negative samples: 2 dialogues mixed by vertical split
    strategy = 'vertical'
    X_path = './%s/words_%s_X.npy' % (sample, strategy)
    positive, negative = tester.load_data(X_path)
    vertical_split_results = tester.test_model(negative)
    print vertical_split_results
