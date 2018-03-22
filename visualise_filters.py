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

import pylab as pl
import matplotlib.cm as cm
# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

from preprocess import X_path, y_path


embeddings_name = 'DBpedia_GlobalVectors_9_pageRank'
# the name of the layer we want to visualize
layer_name = 'conv1d_1'  # 250 filters in this layer (None, 113, 250)
filter_index = 0

with open("./models/%s_model_127932.json" % embeddings_name, 'r') as json_file:
    loaded_model_json = json_file.read()

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

# # get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# print layer_dict


# adopted from https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
# load dataset
data = np.load(X_path)
labels = np.load(y_path)

embedding_matrix = np.load(embeddings['matrix_path'])

# pick a random dialogue sample
i = 4600
# Visualize the first layer of convolutions on an input image
sample = data[i:i+1]
y = labels[i:i+1]
print sample, y

# embed entities
X = [[embedding_matrix[entity_id] for entity_id in dialogue] for dialogue in sample ]


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

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
# layer_output = layer_dict[layer_name].output
# loss = K.mean(layer_output[:, :, filter_index])

# # compute the gradient of the input picture wrt this loss
# grads = K.gradients(loss, input_img)[0]

# # normalization trick: we normalize the gradient
# # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# # this function returns the loss and grads given the input picture
# iterate = K.function([input_img], [loss, grads])

# def normalize(x):
#     # utility function to normalize a tensor by its L2 norm
#     return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


# kept_filters = []
# for filter_index in range(200):
#     # we only scan through the first 200 filters,
#     # but there are actually 512 of them
#     print('Processing filter %d' % filter_index)
#     start_time = time.time()

#     # we build a loss function that maximizes the activation
#     # of the nth filter of the layer considered
#     layer_output = layer_dict[layer_name].output
#     if K.image_data_format() == 'channels_first':
#         loss = K.mean(layer_output[:, filter_index, :])
#     else:
#         loss = K.mean(layer_output[:, :, filter_index])

#     # we compute the gradient of the input picture wrt this loss
#     grads = K.gradients(loss, input_img)[0]
#     print grads

#     # normalization trick: we normalize the gradient
#     grads = normalize(grads)

#     # this function returns the loss and grads given the input picture
#     iterate = K.function([input_img], [loss, grads])

#     # step size for gradient ascent
#     step = 1.

#     # we start from a gray image with some random noise
#     if K.image_data_format() == 'channels_first':
#         input_img_data = np.random.random((1, 3, img_width, img_height))
#     else:
#         input_img_data = np.random.random((1, img_width, img_height, 3))
#     input_img_data = (input_img_data - 0.5) * 20 + 128

#     # we run gradient ascent for 20 steps
#     for i in range(20):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step

#         print('Current loss value:', loss_value)
#         if loss_value <= 0.:
#             # some filters get stuck to 0, we can skip them
#             break

#     # decode the resulting input image
#     if loss_value > 0:
#         img = deprocess_image(input_img_data[0])
#         kept_filters.append((img, loss_value))
#     end_time = time.time()
#     print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# # we will stich the best 64 filters on a 8 x 8 grid.
# n = 8

# # the filters that have the highest loss are assumed to be better-looking.
# # we will only keep the top 64 filters.
# kept_filters.sort(key=lambda x: x[1], reverse=True)
# kept_filters = kept_filters[:n * n]

# # build a black picture with enough space for
# # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
# margin = 5
# width = n * img_width + (n - 1) * margin
# height = n * img_height + (n - 1) * margin
# stitched_filters = np.zeros((width, height, 3))

# # fill the picture with our saved filters
# for i in range(n):
#     for j in range(n):
#         img, loss = kept_filters[i * n + j]
#         stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
#                          (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# # save the result to disk
# imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
