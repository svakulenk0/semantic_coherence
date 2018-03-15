# -*- coding: utf-8 -*-
'''
svakulenko
15 Mar 2018

LSTM neural network for next utterance classification (NUC) task
'''
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# layers used
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# network size params
maxlen = 5
num_features = 5
target_chars = [0, 1]

# input data
# Generate arrays of ints
n_samples = 3
X = np.random.randint(5, size=(n_samples, 5, 5))
y_a = np.random.randint(5, size=(n_samples, 2))
y_t = np.random.randint(5, size=(n_samples, 1))

print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=0.2)(main_input) # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', init='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, init='glorot_uniform', name='time_output')(b2_2)

model = Model(input=[main_input], output=[act_output, time_output])

opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
path_to_model = 'models/model_{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint = ModelCheckpoint(path_to_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

model.fit(X, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, nb_epoch=500)
