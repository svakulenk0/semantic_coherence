# Semantic Coherence

of a conversation or a natural language text in general. Mapping the sequence of entities occuring in the text on the knowledge graph and examining the relations.

# Requirements

* unicodecsv
* spotlight (pip install pyspotlight)



# Run

* Specify path to the input matrices X y and the embeddings matrix:

preprocess.py:

X_path = 'ubuntu127932_X.npy'
y_path = 'ubuntu127932_y.npy'

# embeddings params
embeddings = {
                'DBpedia_GlobalVectors_9_pageRank': {'matrix_path': 'embedding_matrix_DBpedia_GloVe_9PR.npy', 'dims' : 200,
                'all_path': './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'},
                
                'word2vec': {'matrix_path': 'embedding_matrix_word2vec.npy', 'dims' : 300,
                'all_path': './embeddings/GoogleNews-vectors-negative300.bin'},
                
                'GloVe': {'matrix_path': 'embedding_matrix_GloVe.npy', 'dims' : 300,
                'all_path': './embeddings/glove.840B.300d.txt'}
             }

* Load embeddings for the entities in the vocabulary:

preprocess.py: populate_emb_matrix_from_file(embeddings['DBpedia_GlobalVectors']['9_pageRank'])

* Train CNN model:

Point 'embeddings_name' to the embeddings configuration in the 'embeddings' dictionary, e.g. 'DBpedia_GlobalVectors_9_pageRank'

train_model.py



# Experiments


# Approach



## Annotation

### Entity linking

The dialogues are annotated using the dbpedia-spotlight API at http://model.dbpedia-spotlight.org/en/annotate with DBpedia entities e.g. http://dbpedia.org/page/Organisation_of_Islamic_Cooperation

Documentation: http://www.dbpedia-spotlight.org/api

(run with annotate_ubuntu_dialogs() from process_ubuntu_dialogues.py)

2016-10 is the latest version of DBpedia
http://downloads.dbpedia.org/2016-10/


### Entity relations

Topk shortest path HDT web service: http://svhdt.ai.wu.ac.at/dbpedia/query

Front-end: http://svhdt.ai.wu.ac.at/control-panel.tpl



## Embeddings

* Pre-trained RDF2vec embeddings:

data.dws.informatik.uni-mannheim.de/rdf2vec/models

trained on the English version of DBpedia 2016-04
http://downloads.dbpedia.org/2016-04/

* Training

Download the english DBpedia dumps
./download_dbpedia.sh 3.9 en


## Results

Positive example: subgraph matrix of the shortest path between entities

# unique entities mentioned in the dialogue 3
[u'Ubuntu_(philosophy)', u'Intel_80386', u'Web_server']
# paths 3
Graph size: 7 entities 7 edges
{u'Web_server': 5, u'VMware_ESX': 2, u'Ubuntu_(philosophy)': 0, u'Intel_80386': 3, u'Ubuntu_(operating_system)': 1, u'Debian': 4, u'Embedded_system': 6}
[[ 0.  1.  0.  2.  1.  0.  0.]
 [ 0.  0.  1.  0.  0.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  2.  1.]
 [ 0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  1.  0.]]


# Acknowledgments

* https://github.com/botcs/text_cnn
