# -*- coding: utf-8 -*-
'''
svakulenko
3 Apr 2018

Load pre-trained CNN model and visualise filters
'''
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prepare_dataset import load_vocabulary

# in test phase
K.set_learning_phase(0)

SAMPLE = '291848'

positives = SAMPLE + '/words/positive_X.npy'
negatives = SAMPLE + '/words/random_X.npy'
model_weights = SAMPLE + '/words/models/291848_random_GloVe.h5'
model_architecture = SAMPLE + '/words/models/291848_random_GloVe_model.json'
input_length = 128

# load model
with open(model_architecture, 'r') as json_file:
    loaded_model_json = json_file.read()
# print loaded_model_json
model = keras.models.model_from_json(loaded_model_json)

# load pre-trained model weights
model.load_weights(model_weights)
print('Model loaded.')

# load data sample
i = 4608
positive = np.load(positives)[i]
negative = np.load(negatives)[i]
n_words = len(positive)
assert n_words == len(negative)

vocabulary = load_vocabulary(SAMPLE + '/words/vocab.pkl')
inv_vocabulary = {v: k for k, v in vocabulary.iteritems()}

labels = [[''], ['']]
labels[0].extend([inv_vocabulary[_id].encode('utf-8') for _id in positive])
labels[1].extend([inv_vocabulary[_id].encode('utf-8') for _id in negative])
print labels

data = [positive, negative]
print data
data = pad_sequences(data, padding='post', maxlen=input_length)
print model.predict(data)

# from https://github.com/jacobgil/keras-cam/blob/master/cam.py
# print model.layers
class_weights = model.layers[-2].get_weights()[0]
get_output = K.function([model.layers[0].input], [model.layers[2].output, model.layers[-2].output])
[conv_outputs, predictions] = get_output([data])

# from https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
# fig, axes = pl.subplots(nrows=2, ncols=1)
# from https://stackoverflow.com/questions/24535393/matplotlib-getting-subplots-to-fill-figure
fig, axes = pl.subplots(2, 1, figsize=(10,5))  

for j, ax in enumerate(axes.flatten()):  # flatten in case you have a second row at some point
    sample = conv_outputs[j, :, :][:n_words]
    img = ax.imshow(np.squeeze(sample[:]), vmin=0, vmax=6, interpolation='nearest', cmap='RdBu')
    ax.set_aspect('auto')
    ax.set_yticklabels(labels[j])

# pl.colorbar(img)

fig.subplots_adjust(right=0.9)
# left, bottom, width, height
cbar_ax = fig.add_axes([0.92, 0.1, 0.05, 0.8])
fig.colorbar(img, cax=cbar_ax)

pl.show()

# pl.savefig('foo%d.pdf' % i)
