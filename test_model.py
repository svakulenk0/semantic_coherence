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

    def __init__(self, sample, embeddings_name):
        model_path = './models/%s/%s_model.json' % (sample, embeddings_name)
        weights_path = './models/%s/%s.h5' % (sample, embeddings_name)
        vocabulary_path = './%s/vocab_words.pkl' % sample
        self.model = self.load_model(model_path, weights_path)

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

    def test_model(self, X, limit=10):
        '''
        3. Run model (inference)

        limit <int> predict only the first n samples in each dataset
        '''
        return self.model.predict(X[:limit])


if __name__ == '__main__':
    # The best word model obtained by training on negative samples drawn uniformly at random (acc: 98.4 on its test set)
    sample = 'sample172098'
    embeddings_name = 'GloVe'
    tester = Tester(sample, embeddings_name)

    # 1) True positive samples: all real dialogues from the Ubuntu dataset
    strategy = 'random'
    X_path = './%s/words_%s_X.npy' % (sample, strategy)
    positive, negative = tester.load_data(X_path)
    positive_results = tester.test_model(positive)
    print positive_results

    # 2) True negative samples: drawn uniformly at random from the vocabulary
    random_negative_results = tester.test_model(negative)
    print random_negative_results

    # 3) True negative samples: drawn from the vocabulary frequency (count) distribution
    # 4) True negative samples: 2 dialogues mixed by horizontal split
    # 5) True negative samples: 2 dialogues mixed by vertical split


