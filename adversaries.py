# -*- coding: utf-8 -*-
'''
svakulenko
31 Mar 2018

Negative sampling strategies to produce adversary samples
from true positive samples
'''
import numpy as np
import random

from prepare_dataset import load_vocabulary
from prepare_dataset import LATEST_SAMPLE


def generate_uniform_random(folder, sample=LATEST_SAMPLE, test='test/'):
    # load vocabulary
    vocabulary = load_vocabulary('./%s/%s/vocab.pkl' % (sample, folder))
    # load positive samples
    positives = np.load('./%s/%s/%spositive_X.npy' % (sample, folder, test))

    adversaries = []
    
    for dialogue in positives:
        adversaries.append(random.sample(xrange(0, len(vocabulary)), len(dialogue)))

    assert len(adversaries) == len(positives)
    np.save('./%s/%s/%srandom_X.npy' % (sample, folder, test), adversaries)


def generate_adversaries():
    for folder in ['entities', 'words']:
        # development
        generate_uniform_random(folder, test='')
        # test set
        generate_uniform_random(folder)


if __name__ == '__main__':
    generate_adversaries()
