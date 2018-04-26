# -*- coding: utf-8 -*-
'''
svakulenko
23 Mar 2018

Embeddings variable holding paths
'''

# DBpedia vector space embeddings
entity_embeddings = {
                        'GlobalVectors': { '9_pageRank': {'dims' : 200, 'path': './embeddings/DBpediaVecotrs200_20Shuffle.txt'},
                        },

                        'rdf2vec': { '9_pageRank': {'dims' : 200, 'path': './embeddings/biasedRDF2Vec/PageRank/db2vec_sg_200_5_25_5'},
                        },

                     }


# word vector space embeddings
word_embeddings = {
                    'GloVe': {'dims' : 300, 'path': './embeddings/glove.840B.300d.txt'},

                    'word2vec': {'dims' : 300, 'path': './embeddings/GoogleNews-vectors-negative300.bin'}

                 }
