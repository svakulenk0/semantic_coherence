# Semantic Coherence


Vakulenko, S.; de Rijke, M.; Cochez, M.; Savenkov, V.; Polleres, A. (2018). Measuring Semantic Coherence of a Conversation. In: International Semantic Web Conference 2018, Monterey, CA, USA https://arxiv.org/pdf/1806.06411.pdf


## Requirements

* Python 2
* unicodecsv
* spotlight (pip install pyspotlight)
* ...

```
virtualenv myvenv

source myvenv/bin/activate

pip install -r requirements.txt
```


## Run

* prepare_dataset.py: create vocabulary and encode development and training data.
* adversaries.py: generate adversaries
* load_embeddings.py
* train_model.py



Summary: paths to the input matrices and embeddings are specified in preprocess.py. 1) Generate embeddings matrix preprocess.py: populate_emb_matrix_from_file; 2) train_model.py

* Specify path to the input matrices X y and the embeddings matrix:

preprocess.py:

```python
X_path = 'ubuntu127932_X.npy'
y_path = 'ubuntu127932_y.npy'

embeddings = {
                'DBpedia_GlobalVectors_9_pageRank': {'matrix_path': 'embedding_matrix_DBpedia_GloVe_9PR.npy', 'dims' : 200,
                'all_path': './embeddings/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/9_pageRank/DBpediaVecotrs200_20Shuffle.txt'},
                
                'word2vec': {'matrix_path': 'embedding_matrix_word2vec.npy', 'dims' : 300,
                'all_path': './embeddings/GoogleNews-vectors-negative300.bin'},
                
                'GloVe': {'matrix_path': 'embedding_matrix_GloVe.npy', 'dims' : 300,
                'all_path': './embeddings/glove.840B.300d.txt'}
             }
```

1. Load embeddings for the entities in the vocabulary:

preprocess.py: populate_emb_matrix_from_file(embeddings['DBpedia_GlobalVectors_9_pageRank'])

2. Train CNN model:

Point 'embeddings_name' to the embeddings configuration in the 'embeddings' dictionary, e.g. 'DBpedia_GlobalVectors_9_pageRank'

train_model.py



## Annotation

### Entity linking

The dialogues are annotated using the dbpedia-spotlight API at http://model.dbpedia-spotlight.org/en/annotate with DBpedia entities e.g. http://dbpedia.org/page/Organisation_of_Islamic_Cooperation

Documentation: http://www.dbpedia-spotlight.org/api

(run with annotate_ubuntu_dialogs() from process_ubuntu_dialogues.py)

2016-10 is the latest version of DBpedia
http://downloads.dbpedia.org/2016-10/


## Embeddings

* Pre-trained RDF2vec and KGlove embeddings:

data.dws.informatik.uni-mannheim.de/rdf2vec/models

trained on the English version of DBpedia 2016-04
http://downloads.dbpedia.org/2016-04/

## Acknowledgment

This work is supported by the project 855407 “Open Data for Local Communities” (CommuniData) of the Austrian Federal Ministry of Transport, Innovation and Technology (BMVIT) under the program “ICT of the Future.” Svitlana Vakulenko was supported by the EU H2020 programme under the MSCA-RISE agreement 645751 (RISE BPM).
