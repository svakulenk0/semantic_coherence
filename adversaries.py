# -*- coding: utf-8 -*-
'''
svakulenko
31 Mar 2018

Different negative sampling strategies to produce adversary samples by corrupting true positive samples
'''
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

from prepare_dataset import load_vocabulary
from prepare_dataset import LATEST_SAMPLE
from sample291848 import entity_distribution, word_distribution


def generate_uniform_random(folder, sample=LATEST_SAMPLE, test='test/', **kwargs):
    '''
    Pick items from the vocabulry unifromly at random using the same length as of a positive example
    '''
    # load vocabulary
    vocabulary = load_vocabulary('./%s/%s/vocab.pkl' % (sample, folder))
    # load positive samples
    positives = np.load('./%s/%s/%spositive_X.npy' % (sample, folder, test))

    adversaries = []
    
    for dialogue in positives:
        adversaries.append(random.sample(xrange(0, len(vocabulary)), len(dialogue)))

    assert len(adversaries) == len(positives)
    np.save('./%s/%s/%srandom_X.npy' % (sample, folder, test), adversaries)


def generate_vocabulary_distribution(folder, vocabulary_distribution, sample=LATEST_SAMPLE, test='test/'):
    '''
    Pick items from the vocabulry unifromly at random using the same length as of a positive example
    '''
    # load vocabulary
    vocabulary = load_vocabulary('./%s/%s/vocab.pkl' % (sample, folder))
    # load positive samples
    positives = np.load('./%s/%s/%spositive_X.npy' % (sample, folder, test))

    adversaries = []

    # prepare probabilities from vocabulary counts distribution
    entities = vocabulary_distribution.keys()
    entities_counts = vocabulary_distribution.values()
    entities_probs = [count / float(sum(entities_counts)) for count in entities_counts]

    
    for dialogue in positives:
        adversary = np.random.choice(entities, replace=False, size=len(dialogue), p=entities_probs)
        adversaries.append(adversary)

    assert len(adversaries) == len(positives)
    np.save('./%s/%s/%sdistribution_X.npy' % (sample, folder, test), adversaries)


def generate_sequence_disorder(folder, sample=LATEST_SAMPLE, test='test/', **kwargs):
    '''
    Randomly rearrange (permute) the original sequence (sentence ordering task)
    '''
    # load positive samples
    positives = np.load('./%s/%s/%spositive_X.npy' % (sample, folder, test))

    adversaries = []
    
    for dialogue in positives:
        # randomly permute list of ids
        random.shuffle(dialogue)
        adversaries.append(dialogue)

    assert len(adversaries) == len(positives)
    np.save('./%s/%s/%sdisorder_X.npy' % (sample, folder, test), adversaries)


def merge_horizontally(dialogue1, dialogue2):
    '''
    merge the first half of the first dialogue with the second part of the second dialogue
    '''
    adversary = dialogue1[:len(dialogue1)/2]
    adversary.extend(dialogue2[len(dialogue2)/2:])
    return adversary


# def generate_vertical_split():
def generate_horizontal_split(folder, sample=LATEST_SAMPLE, test='test/', **kwargs):
    # load positive samples
    positives = np.load('./%s/%s/%spositive_X.npy' % (sample, folder, test))
    # chain head to the tail
    adversaries = [ merge_horizontally(positives[0], positives[-1]) ]
    
    for i, dialogue in enumerate(positives[1:]):
        # merge previous dialogue with the current dialogue
        adversary = merge_horizontally(positives[i-1], dialogue) 
        adversaries.append(dialogue)

    assert len(adversaries) == len(positives)
    np.save('./%s/%s/%sdisorder_X.npy' % (sample, folder, test), adversaries)


def generate_adversaries(generator):
    vocabulary_distributions = [entity_distribution, word_distribution]
    for i, folder in enumerate(['entities', 'words']):
        # development
        generator(folder, test='', vocabulary_distribution=vocabulary_distributions[i])
        # test set
        generator(folder, vocabulary_distribution=vocabulary_distributions[i])


if __name__ == '__main__':
    generate_adversaries(generate_horizontal_split)
