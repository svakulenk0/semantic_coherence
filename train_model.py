# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Load and split the dataset to train the classification model
'''
import numpy as np

from model import train
from preprocess import X_path, y_path, embeddings

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

# dataset params
vocabulary_size = 19660  # unique entities + extra token 0 for UNK
# input_length = 254

# load dataset
data = np.load(X_path)
# print X.shape[0], X.shape[1]
labels = np.load(y_path)
# print y.shape[0]

# labels = np.full((X.shape[0]), 1)
input_length = data.shape[1]
print 'max input length:', input_length


# split the data into a test set and a training set
# https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(TEST_SPLIT * data.shape[0])

x = data[:-num_validation_samples]
y = labels[:-num_validation_samples]
X_test = data[-num_validation_samples:]
y_test = labels[-num_validation_samples:]

# split the training set into a training set and a validation set
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
num_validation_samples = int(VALIDATION_SPLIT * x.shape[0])

x_train = x[:-num_validation_samples]
y_train = y[:-num_validation_samples]
x_val = x[-num_validation_samples:]
y_val = y[-num_validation_samples:]

embeddings_name = 'DBpedia_GlobalVectors_9_pageRank'

model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings[embeddings_name])
# train(X, y, X, y, vocabulary_size, input_length, embeddings['GloVe'])

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy * 100))

# Save the trained model
# serialize model to JSON
model_json = model.to_json()
with open("./models/%s_model_127932.json" % embeddings_name, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./models/%s_weights_127932.h5' % embeddings_name)
print("Saved model to disk")
