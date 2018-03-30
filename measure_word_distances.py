# -*- coding: utf-8 -*-
'''
svakulenko
30 Mar 2018

Estimate pair-wise distances between words in the conversation
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import word_embeddings

SAMPLE_WORDS_4606 = [u'zip', u'file', u'.tar.gz', u'md5sum', u'md5', u'ubuntu', u'OSX', u'computer', u'Leopard', u'apt-get', u'runlevel', u'bluetooth', u'init', u'gdm', u'rc2', u'spawning', u'walkthrough', u'battle', u'compiz']


def measure_word_distances(sample=SAMPLE_WORDS_4606):
    # load embeddings
    embeddings = {}

    with open(word_embeddings['GloVe']['path']) as embs_file:
        for line in embs_file:
            wordAndVector = line.split(None, 1)
            word = wordAndVector[0]
            # collect embeddings for the words in the sample dialogue
            if word in sample:
                vector = wordAndVector[1]
                vector = vector.split()
                embedding_vector = np.asarray(vector, dtype='float32')
                embeddings[word] = embedding_vector
                if len(embeddings) >= len(sample):
                    print "Found embeddings for all words in the sample"
    print len(embeddings), 'embeddings loaded for ', len(sample), 'words in the sample dialogue'
    print embeddings.keys()

    # snowball
    previous_word_vectors = [[]]
    # and store distances (cosine similarities) between preceding words
    words_distances = []
    
    for word in sample:
        print word
        word_vector = embeddings[word]
        # estimate distances from new word to all previous words
        # compare with cosine between the new word vector and the word vectors of the previous words
        if previous_word_vectors:
            word_distances = cosine_similarity([word_vector], previous_word_vectors)
        previous_word_vectors.append(word_vector)
        words_distances.append(word_distances)
    
    print words_distances


if __name__ == '__main__':
    measure_word_distances()
