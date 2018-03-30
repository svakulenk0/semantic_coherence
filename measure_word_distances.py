# -*- coding: utf-8 -*-
'''
svakulenko
30 Mar 2018

Estimate pair-wise distances between words in the conversation
'''
import gensim
from sklearn.metrics.pairwise import cosine_similarity

from annotate_shortest_paths import SAMPLE_WORDS_4606 as SAMPLE_WORDS
from embeddings import word_embeddings


def measure_word_distances(sample=SAMPLE_WORDS):
    # load embeddings
    model = gensim.models.Word2Vec.load(word_embeddings['GloVe']['path'])
    
    # snowball
    previous_words = {}
    # and store distances (cosine similarities) between preceding words
    words_distances = []
    
    for word in sample:
        word_distances = []
        print word
        # get word vector
        word_vector = model.wv[word]

        # estimate distances from new word to all previous words:
        # iterate over the vectors of the previous words in the conversation
        for previous_word_vector in previous_words.values():
            # cosine between the two vectors
            print word, previous_word
            word_distances.append(cosine_similarity(word_vector, previous_word_vector))

        previous_words[word] = word_vector

        words_distances.append(word_distances)
    
    print previous_words
    print words_distances


if __name__ == '__main__':
    measure_word_distances()
