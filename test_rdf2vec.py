'''
Created on Mar 18, 2016

@author: svakulenko
'''
from numpy import asarray

# 'DB2Vec_sg_500_5_5_15_2_500'
DBPEDIA_EMBS = './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2015-10/noTypes/db2vec_sg_200_5_25_5'


def load_iteratively(emb_path=DBPEDIA_EMBS):
    # from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    embeddings_index = dict()
    f = open(emb_path)
    for line in f:
        values = line.split()
        word = values[0]
        vector = asarray(values[1:], dtype='float32')
        print word
        print vector
        return
        embeddings_index[word] = vector
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))


def load_with_gensim(emb_path=DBPEDIA_EMBS):
    # loads the whole embeddings model into memory
    import gensim
    model = gensim.models.Word2Vec.load(emb_path)
    print ('dbr:Rocky')
    print (model.most_similar(positive=['dbr:Rocky'], topn=100))


if __name__ == '__main__':
    load_iteratively()
