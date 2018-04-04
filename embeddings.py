# -*- coding: utf-8 -*-
'''
svakulenko
23 Mar 2018

Embeddings variable holding paths
'''

# DBpedia vector space embeddings
# DBpedia vector space embeddings
entity_embeddings = {
'GlobalVectors': { 
        '1_uniform': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/1_uniform/DBpediaVecotrs200_20Shuffle.txt'},
        '2_predicate': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/2_predicate/DBpediaVecotrs200_20Shuffle.txt'},
        '3_inversePredicate': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/3_inversePredicate/DBpediaVecotrs200_20Shuffle.txt'},
        '4_predicateObject': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/4_predicateObject/DBpediaVecotrs200_20Shuffle.txt'},
        '5_inversePredicateObject': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/5_inversePredicateObject/DBpediaVecotrs200_20Shuffle.txt'},
        '6_object': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/6_object/DBpediaVecotrs200_20Shuffle.txt'},
        '7_inverseObject': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/7_inverseObject/DBpediaVecotrs200_20Shuffle.txt'},
        '8_InverseObjectSplit': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/8_InverseObjectSplit/DBpediaVecotrs200_20Shuffle.txt'},
        '9_pageRank': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'},
        '10_inversePageRank': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/10_inversePageRank/DBpediaVecotrs200_20Shuffle.txt'},
        '11_pageRankSplit': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/11_pageRankSplit/DBpediaVecotrs200_20Shuffle.txt'},
        '12_inversePageRankSplit': {'dims' : 200, 'path': '/home/cochez/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/12_inversePageRankSplit/DBpediaVecotrs200_20Shuffle.txt'},
     #   '9_pageRank_no_context_combine': {'dims' : 200, 'path': '/home/cochez/bigRunsGloveEmbedding/vectors.100.txt'},

    },

    'rdf2vec': { 
        'InversePredicateFrequency': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/InversePredicateFrequency/db2vec_sg_200_5_25_5'},
        'PageRank': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/PageRank/db2vec_sg_200_5_25_5'},
        'PageRankSplit': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/PageRankSplit/db2vec_sg_200_5_25_5'},
        'predicateFrequency': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/predicateFrequency/db2vec_sg_200_5_25_5'},
#        'RDF2vecGlove': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/RDF2vecGlove/db2vec_sg_200_5_25_5'},
        'uniform': {'dims' : 200, 'path': '/home/cochez//biasedRDF2VecRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/Biased/uniform/db2vec_sg_200_5_25_5'},
    }
}


# word vector space embeddings
word_embeddings = {
                    'GloVe': {'dims' : 300, 'path': './embeddings/glove.840B.300d.txt'},

                    'word2vec': {'dims' : 300, 'path': './embeddings/GoogleNews-vectors-negative300.bin'}

                 }
