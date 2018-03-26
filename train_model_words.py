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
epochs = 5

# dataset params
sample = 'sample172098'
vocabulary_size = 30541  # unique entities + extra token 0 for UNK
X_path_words = './%s/words_X.npy' % sample
y_path = './%s/y.npy' % sample

x_train, y_train, x_val, y_val, x_test, y_test, input_length = load_dataset_splits(X_path_words, y_path, test_split=0.2, validation_split=0.2)

for label, embeddings_config in word_embeddings.items():
    print label
    embeddings_config['matrix_path'] = PATH + label + '.npy'
    model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings_config, label, batch_size, epochs)

    # evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))

    # serialize the trained model to JSON
    model_json = model.to_json()
    with open("./models/%s_model_172098.json" % label, "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights('./models/%s_weights_172098.h5' % label)
    print("Saved model to disk")
