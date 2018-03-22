# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Load pre-trained CNN model and visualise filters
'''
import time
import numpy as np

# import theano
import keras
from keras import backend as K
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.cm as cm
import cv2
# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

from preprocess import X_path, y_path, embeddings

np.random.seed(1337) # for reproducibility
# K._LEARNING_PHASE = tf.constant(0) # test mode
K.set_learning_phase(1)

embeddings_name = 'DBpedia_GlobalVectors_9_pageRank'
# the name of the layer we want to visualize
conv_layer = 'conv1d_1'  # 250 filters in this layer (None, 113, 250)
filter_index = 0
output_path = "heatmap.jpg"


with open("./models/%s_model_127932.json" % embeddings_name, 'r') as json_file:
    loaded_model_json = json_file.read()
# print loaded_model_json
model = keras.models.model_from_json(loaded_model_json)

# load pre-trained model weights
model.load_weights('./models/%s_weights_127932.h5' % embeddings_name)
print('Model loaded.')

# model.summary()

# https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
# convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))

# adopted from https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
# # this is the placeholder for the input images
# input_img = model.input
# print input_img

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print layer_dict


# load dataset
data = np.load(X_path)
labels = np.load(y_path)


# pick a postitive dialogue sample: even are positive samples, odd are negative samples
i = 4606
n_samples = 1
# Visualize the first layer of convolutions on an input image
sample = data[i:i+n_samples]
# y = labels[i:i+1]
width, height = sample.shape
# print sample, y
print 'input shape:', sample.shape

print(model.predict(sample))


# embed entities
embedding_matrix = np.load(embeddings[embeddings_name]['matrix_path'])
X = [[embedding_matrix[entity_id] for entity_id in dialogue if entity_id != 0] for dialogue in sample ]
print X

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

pl.figure()
pl.title('input')
nice_imshow(pl.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)
pl.savefig('foo.pdf')
