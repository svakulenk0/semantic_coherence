# -*- coding: utf-8 -*-
'''
svakulenko
31 Mar 2018

Different negative sampling strategies to produce adversary samples by corrupting true positive samples
'''
import numpy as np
import random
import json
from heapq import heappush, heappop

from keras.preprocessing.sequence import pad_sequences

from prepare_dataset import load_vocabulary, VOCAB_ENTITIES_PATH, VOCAB_WORDS_PATH
from prepare_dataset import LATEST_SAMPLE, DEV_DATA_PATH, TEST_DATA_PATH
from sample291848 import entity_distribution, word_distribution
from annotate_ubuntu_dataset import ANNOTATION_FILE


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


def merge_verticaly(dialogue1, dialogue2, entity_vocabulary, word_vocabulary):
    turns1, entities1, words1 = dialogue1
    turns2, entities2, words2 = dialogue2

    entity_adversary, words_adversary = [], []
    j = 0
    for i, turn in enumerate(turns1):
        if turn == 0:
            entity = entity_vocabulary[entities1[i]]
            if entity not in entity_adversary:
                entity_adversary.append(entity)
                words_adversary.extend([word_vocabulary[word] for word in words1[i].split()])
        else:
            for turn2 in turns2[j:]:
                j += 1
                if turn2 == 1:
                    entity = entity_vocabulary[entities2[j-1]]
                    if entity not in entity_adversary:
                        entity_adversary.append(entity)
                        words_adversary.extend([word_vocabulary[word] for word in words2[j-1].split()])
                        break

    return entity_adversary, words_adversary


def generate_vertical_split(sample=LATEST_SAMPLE):
    '''
    '''
    entity_vocabulary = load_vocabulary(VOCAB_ENTITIES_PATH % sample)
    word_vocabulary = load_vocabulary(VOCAB_WORDS_PATH % sample)

    test = ['', 'test/']
    entities_adversaries, words_adversaries = [], []
    # , TEST_DATA_PATH
    for i, path in enumerate([DEV_DATA_PATH]):
        dialogues_by_length = []
        
        # load dialogues and order them by length
        with open(path, "rb") as entities_file:
            for line in entities_file:
                annotation = json.loads(line)
                heappush(dialogues_by_length, (annotation['turns'], annotation['entity_URIs'], annotation['surface_forms']))
        
        # iterate over the dialogues
        while len(dialogues_by_length) > 1:
            # get a pair of dialogues with similar length
            dialogue1 = heappop(dialogues_by_length)
            dialogue2 = heappop(dialogues_by_length)

            entity_adversary1, words_adversary1 = merge_verticaly(dialogue1, dialogue2, entity_vocabulary, word_vocabulary)
            entity_adversary2, words_adversary2 = merge_verticaly(dialogue2, dialogue1, entity_vocabulary, word_vocabulary)
            print words_adversary1
            print words_adversary2
            entities_adversaries.extend([entity_adversary1, entity_adversary2])
            words_adversaries.extend([words_adversary1, words_adversary2])
            
        np.save('./%s/entities/%svertical_X.npy' % (sample, test[i]), entities_adversaries)
        np.save('./%s/words/%svertical_X.npy' % (sample, test[i]), words_adversaries)


def merge_horizontally(dialogue1, dialogue2):
    '''
    merge the first half of the first dialogue with the second part of the second dialogue
    '''
    adversary = dialogue1[:len(dialogue1)/2]
    # control for duplicates!
    adversary.extend([item for item in dialogue2[len(dialogue2)/2:] if item not in adversary])
    return adversary


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
    np.save('./%s/%s/%shorizontal_X.npy' % (sample, folder, test), adversaries)


def generate_adversaries(generator):
    vocabulary_distributions = [entity_distribution, word_distribution]
    for i, folder in enumerate(['entities', 'words']):
        # development
        generator(folder, test='', vocabulary_distribution=vocabulary_distributions[i])
        # test set
        generator(folder, vocabulary_distribution=vocabulary_distributions[i])


if __name__ == '__main__':
    generate_adversaries(generate_sequence_disorder)
    generate_adversaries(generate_horizontal_split)
    generate_vertical_split()
