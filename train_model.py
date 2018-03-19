# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Train CNN to classify dialogues using DBpedia entity annotations as input
'''
import numpy as np

from process_ubuntu_dialogues import load_annotated_dialogues, load_vocabulary
from model import train

vocabulary = load_vocabulary()

# load correct and incorrect examples
X, labels = load_annotated_dialogues(vocabulary)
# labels = np.full((X.shape[0]), 1)
input_length = X.shape[1]
print labels, 'max input length:', input_length

train(X, labels, X, labels, vocabulary, input_length)
