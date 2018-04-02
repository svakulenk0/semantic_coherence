# -*- coding: utf-8 -*-
'''
svakulenko
30 Mar 2018

Estimate pair-wise distances between words in the conversation

loaded embeddings for all words (but had to manually convert '.tar.gz' to 'tar.gz')
'''
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from embeddings import word_embeddings

SAMPLE_WORDS_4606 = [u'zip', u'file', u'tar.gz', u'md5sum', u'md5', u'ubuntu', u'OSX', u'computer', u'Leopard', u'apt-get', u'runlevel', u'bluetooth', u'init', u'gdm', u'rc2', u'spawning', u'walkthrough', u'battle', u'compiz']

cosines = [[[0.9999998807907104]], [[1.0, 0.12084029614925385]], [[1.0, 0.12084029614925385, 0.24155808985233307]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019557237625, 0.33175957202911377]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019557237625, 0.33175957202911377, 0.1760643720626831]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648636102676392]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.4661942720413208]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.4661943018436432, 0.5392439365386963]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.4661943018436432, 0.5392439365386963, 0.1377360224723816]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.466194212436676, 0.5392439961433411, 0.13773605227470398, 0.15335698425769806]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.466194212436676, 0.5392439961433411, 0.13773605227470398, 0.15335698425769806, -0.01058376207947731]], [[1.0, 0.12084032595157623, 0.24155810475349426, 0.29122525453567505, 0.1974019706249237, 0.33175957202911377, 0.1760643869638443, 0.0212489515542984, 0.12648634612560272, 0.36780887842178345, 1.0, 0.0928681492805481, 0.466194212436676, 0.5392439961433411, 0.13773605227470398, 0.15335698425769806, -0.01058376207947731, -0.026089638471603394]]]


def get_maximum_similarities(sample=SAMPLE_WORDS_4606, entities_cosines=cosines):
    '''
    find maximum similarities (minimum distances) in the sequence of words
    '''
    # print len(sample)
    # print len(cosines)
    # print [np.argmax(enity_cosine) for enity_cosine in entities_cosines]
    return [np.max(enity_cosine) for enity_cosine in entities_cosines]


def load_GloVe_embeddings():
    # load all embeddings in a dictionary
    embeddings = {}
    with open(word_embeddings['GloVe']['path']) as embs_file:
        for line in embs_file:
            wordAndVector = line.split(None, 1)
            word = wordAndVector[0]
            # collect embeddings for the words in the sample dialogue
            # if word in sample:
            vector = wordAndVector[1]
            vector = vector.split()
            embedding_vector = np.asarray(vector, dtype='float32')
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
            embeddings[word] = embedding_vector
            # if len(embeddings) >= len(sample):
            #     print "Found embeddings for all words in the sample"
    print len(embeddings), 'embeddings loaded'  # for ', len(sample), 'words in the sample dialogue'
    # print embeddings.keys()
    return embeddings


def measure_word_distances(embeddings, sample=SAMPLE_WORDS_4606):
    # snowball
    previous_word_vectors = np.array([[]], ndmin=2)
    # and store distances (cosine similarities) between preceding words
    words_distances = []
    
    for word in sample:
        # print word
        if word in embeddings:
            word_vector = embeddings[word]
            # estimate distances from new word to all previous words
            # compare with cosine between the new word vector and the word vectors of the previous words
            if previous_word_vectors.size > 0:
                word_distances = cosine_similarity(word_vector, previous_word_vectors)
                words_distances.append(word_distances.tolist())
                previous_word_vectors = np.append(previous_word_vectors, word_vector, axis=0)
            else:
                # first word in the dialogue
                words_distances.append([])
                previous_word_vectors = word_vector
    return words_distances


def measure_min_distances(embeddings, sample):
    # snowball
    previous_word_vectors = np.array([[]], ndmin=2)
    # and store distances (cosine similarities) between preceding words
    min_words_distances = []
    
    for word in sample:
        # print word
        if word in embeddings:
            word_vector = embeddings[word]
            # estimate distances from new word to all previous words
            # compare with cosine between the new word vector and the word vectors of the previous words
            if previous_word_vectors.size > 0:
                word_distances = cosine_similarity(word_vector, previous_word_vectors).tolist()
                # min distance = max similarity
                min_words_distances.extend([np.max(enity_cosine) for enity_cosine in word_distances])
                previous_word_vectors = np.append(previous_word_vectors, word_vector, axis=0)
            else:
                # first word in the dialogue
                previous_word_vectors = word_vector
    return min_words_distances


def collect_word_distances(embeddings, samples_type, sample='291848'):
    
    # load data
    samples = np.load('./%s/words/%s_X.npy' % (sample, samples_type))
    print samples.shape[0], 'samples'

    # collect distance distributions across dialogues
    words_distances = []
    for positive in positives:
        words_distances.extend(measure_min_distances(embeddings, sample))
    return words_distances


def compare_distance_distributions():
    '''
    compare word distance distributions in dialogues
    '''
    embeddings = load_GloVe_embeddings()

    positive_distances = collect_word_distances('positive', embeddings)
    positive_distances_distribution = Counter(positive_distances)
    print positive_distances_distribution
    
    random_distances = collect_word_distances('random', embeddings)
    random_distances_distribution = Counter(random_distances)
    print random_distances_distribution

    # print stats.entropy(pk=distribution_positives, qk=distribution_random)


if __name__ == '__main__':
    compare_distance_distributions()
