# -*- coding: utf-8 -*-
'''
svakulenko
3 Apr 2018

Load pre-trained CNN model and visualise filters
'''
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

SAMPLE = '291848'

positives = SAMPLE + '/words/positive_X.npy'
negatives = SAMPLE + '/words/random_X.npy'
model_weights = SAMPLE + '/words/models/291848_random_GloVe.h5'
model_architecture = SAMPLE + '/words/models/291848_random_GloVe_model.json'
input_length = 128

# load data sample
i = 4608
n = 1
data = np.load(positives)[i:i+n]
data = pad_sequences(data, padding='post', maxlen=input_length)


# load model
with open(model_architecture, 'r') as json_file:
    loaded_model_json = json_file.read()
# print loaded_model_json
model = keras.models.model_from_json(loaded_model_json)

# load pre-trained model weights
model.load_weights(model_weights)
print('Model loaded.')

print model.predict(data)
