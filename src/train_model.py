# -*- coding: utf-8 -*-
'''
svakulenko
19 Mar 2018

Load and split the dataset to train the classification model
'''
from model import train
from prepare_dataset import load_dataset_splits
from load_embeddings import PATH
from embeddings import entity_embeddings as embeddings

# training parameters:
batch_size = 128
epochs = 5

# dataset params
sample = 'sample172098'
vocabulary_size = 24081  # unique entities + extra token 0 for UNK

# iterate over datasets
negative_sampling_strategies = ['random', 'disorder', 'distribution', 'vertical', 'horizontal']
# negative_sampling_strategies = [ # 'random', 
# 'vertical', 'horizontal']

for strategy in negative_sampling_strategies:

    print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@STRATEGY:" + strategy
    X_path_entities = '../%s/entities_%s_X.npy' % (sample, strategy)
    y_path = '../%s/y.npy' % sample

    x_train, y_train, x_val, y_val, x_test, y_test, input_length = load_dataset_splits(X_path_entities, y_path, test_split=0.2, validation_split=0.2)

    for embedding_model in embeddings:
        for embeddings_name, embeddings_config in embeddings[embedding_model].items():
            label = '%s_%s' % (embedding_model, embeddings_name)
            print label
            embeddings_config['matrix_path'] = PATH + label + '.npy'
            model = train(x_train, y_train, x_val, y_val, vocabulary_size, input_length, embeddings_config, batch_size, epochs)

            # evaluate the model
            loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
            print('Accuracy: %f' % (accuracy * 100))

            # serialize the trained model to JSON
            model_json = model.to_json()
            with open("./models/%s_%s_model_127932.json" % (strategy, label), "w") as json_file:
                json_file.write(model_json)
            
            # serialize weights to HDF5
            model.save_weights('./models/%s_%s_weights_127932.h5' % (strategy,label))
            print("Saved model to disk")

print ("Make sure to rerun the needed parts for the random startegy")
