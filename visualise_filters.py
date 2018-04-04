# -*- coding: utf-8 -*-
'''
svakulenko
3 Apr 2018

Load pre-trained CNN model and visualise filters
'''
import os
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prepare_dataset import load_vocabulary

# in test phase
K.set_learning_phase(0)

# specify sample trained on
SAMPLE = '291848'

# load data sample 4610
i = 4605


def visualise_activations():
    positives = SAMPLE + '/words/positive_X.npy'
    positive = np.load(positives)[i]

    vocabulary = load_vocabulary(SAMPLE + '/words/vocab.pkl')
    inv_vocabulary = {v: k for k, v in vocabulary.iteritems()}


    labels = [[], []]
    labels[0] = [inv_vocabulary[_id].encode('utf-8') for _id in positive]
    print labels[0]


    # specify which trained models to visualise
    # for model_name in ['random', 'distribution', 'disorder', 'horizontal', 'vertical']:
    for model_name in ['distribution']:
        os.mkdir('figs/291848/%s' % model_name)

        model_weights = SAMPLE + '/words/models/291848_%s_GloVe.h5' % model_name
        model_architecture = SAMPLE + '/words/models/291848_%s_GloVe_model.json' % model_name
        input_length = 128

        # load model
        with open(model_architecture, 'r') as json_file:
            loaded_model_json = json_file.read()
        # print loaded_model_json
        model = keras.models.model_from_json(loaded_model_json)

        # load pre-trained model weights
        model.load_weights(model_weights)
        print('Model loaded.')

        # for negative_type in ['random', 'distribution', 'disorder', 'horizontal', 'vertical']:
        for negative_type in ['random', 'horizontal']:
        # for negative_type in ['horizontal']:
            print negative_type

            negatives = SAMPLE + '/words/%s_X.npy' % negative_type
            negative = np.load(negatives)[i]

            n_words = len(positive)
            # assert n_words == len(negative)

            labels[1] = [inv_vocabulary[_id].encode('utf-8') for _id in negative]
            print labels[1]

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
            fig, axes = pl.subplots(2, 1, figsize=(10, 8))

            for j, ax in enumerate(axes.flatten()):  # flatten in case you have a second row at some point
                sample = conv_outputs[j, :, :][:n_words]
                img = ax.imshow(np.squeeze(sample[:]), vmin=0, vmax=6, interpolation='nearest', cmap='RdBu')
                ax.set_aspect('auto')
                ax.set_yticks(np.arange(0, n_words, 1))
                ax.set_yticklabels(labels[j])

            pl.title(negative_type)

            fig.subplots_adjust(right=0.9)
            # left, bottom, width, height
            cbar_ax = fig.add_axes([0.92, 0.1, 0.05, 0.8])
            fig.colorbar(img, cax=cbar_ax)
            
            # pl.show()
            # return

            pl.savefig('figs/291848/%s/heatmap_%s_%d.pdf' % (model_name, negative_type, i))


if __name__ == '__main__':
    visualise_activations()
