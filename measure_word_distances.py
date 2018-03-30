# -*- coding: utf-8 -*-
'''
svakulenko
30 Mar 2018

Estimate pair-wise distances between words in the conversation
'''
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
    
    # snowball
    previous_words = []
    # and store distances (cosine similarities) between preceding words
    words_distances = []
    
    for word in sample:
        word_distances = []
        print word
        # estimate distances from new word to all previous words:
        # iterate over the vectors of the previous words in the conversation
        for previous_word in previous_words:
            # compare with cosine between the two word vectors
            print word, previous_word
            word_distances.append(cosine_similarity(embeddings[word], embeddings[previous_word])

        previous_words.append(word)

        words_distances.append(word_distances)
    
    print previous_words
    print words_distances


if __name__ == '__main__':
    measure_word_distances()
