# -*- coding: utf-8 -*-
'''
svakulenko
25 Mar 2018

Load word embeddings: glove and word2vec
'''
# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Load and split the dataset to train the classification model
'''
from model import train
from prepare_dataset import load_dataset_splits
from load_embeddings import PATH
from embeddings import word_embeddings

# training parameters:
batch_size = 128
epochs = 20

# dataset params
sample = 'sample172098_new'
vocabulary_size = 30541  # unique entities + extra token 0 for UNK
# specify negative sampling strategies e.g. 'random', 'vertical', 'horizontal'
negative_sampling_strategies = ['random']
# specify embeddings, e.g. GloVe, word2vec
embedding_names = ['GloVe']

for strategy in negative_sampling_strategies:
    X_path_words = './%s/words_%s_X.npy' % (sample, strategy)
    y_path = './%s/y.npy' % sample

    x_train, y_train, x_val, y_val, x_test, y_test, input_length = load_dataset_splits(X_path_words, y_path, test_split=0.2, validation_split=0.2)

    for embeddings_name in embedding_names:
        label = "%s_%s" % (strategy, embeddings_name)
        print label
        embeddings_config = word_embeddings[embeddings_name]
        embeddings_config['matrix_path'] = PATH + embeddings_name + '.npy'
        model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings_config, label, batch_size, epochs)

        # evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print('Accuracy: %f' % (accuracy * 100))

        # serialize the trained model to JSON
        model_json = model.to_json()
        with open("./models/%s_%s_model.json" % (sample, label), "w") as json_file:
            json_file.write(model_json)
    
    # # serialize weights to HDF5
    # model.save_weights('./models/%s_weights_172098.h5' % label)
    # print("Saved model to disk")
