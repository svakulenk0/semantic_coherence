# -*- coding: utf-8 -*-
'''
svakulenko
17 Mar 2018

Small sample input data for training the model.
DBpedia resources namespace dbr: for http://dbpedia.org/resource/
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
'''
from numpy import array

from preprocess import preprocess, populate_emb_matrix_from_file

DBPEDIA_GLOBAL_PR = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'

# 1. Input: sample data
# document is a dialogue represented as a set of DBpedia concepts
dialogues = [[u'<http://dbpedia.org/resource/Arch>', u'<http://dbpedia.org/resource/Sudo>',
              u'<http://dbpedia.org/resource/Organisation_of_Islamic_Cooperation>'],
             [u'<http://dbpedia.org/resource/CPU_cache>', u'<http://dbpedia.org/resource/Password>']]

# define class labels
labels = array([0, 1])

# 2. Analyse input data
input_length = 3  # maximum number of concepts per document-dialogue
# vocabulary encodes all unique concepts in the input data with integer ids
vocabulary = {u'<http://dbpedia.org/resource/Arch>': 1,
              u'<http://dbpedia.org/resource/Sudo>': 2,
              u'<http://dbpedia.org/resource/Organisation_of_Islamic_Cooperation>': 3,
              u'<http://dbpedia.org/resource/CPU_cache>': 4,
              u'<http://dbpedia.org/resource/Password>': 5}
# vocab_size =  len(vocabulary.keys())
vocab_size =  6
# test preprocessing input docs
print preprocess(dialogues, vocabulary, input_length)

# 3. Embeddings: load pre-trained entity embeddings reflecting the KG structure for each entity in the vocabulary
# e.g. from global vectors pre-trained embeddings /DBpedia/2016-04/9_pageRank from:
# http://data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt
embeddings_dim = 200
print populate_emb_matrix_from_file(vocabulary, vocab_size, embeddings_dim, emb_path=DBPEDIA_GLOBAL_PR)
