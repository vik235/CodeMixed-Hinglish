import os
import keras
import io
import csv
import model
import dataset
import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Dropout, SimpleRNN, LSTM, Bidirectional
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import TensorBoard
import keras_metrics
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

'''
Model Parameters and script settings:
'''

# Set params for the model
tensorBoard_logs_dir = '.\\logs\\'
model_dir = '.\\models\\'
glove_dir = '.\\models\\glove'
train_dir = '.\\data\\train'
train_data = 'messages_2.csv'
label_data = 'labels_2.csv'

'''
Hyper parameters for the model, training
'''
maxlen = 200              # Maximum length of the message/ tweet to be considered
training_samples = 2500
testing_samples = 3189 - training_samples
max_words = 10000         # vocab length
num_filters = 64
embedding_dim = 100
num_classes = 3
num_hidden_lstm = 64
num_hidden_rnn_final = 32
num_dense_fc = 64
recurrent_dropout = 0.2
fc_dropout = 0.5
cnn_dropout = 0.5
learning_rate = 0.01
lr_decay = 1e-6
patience = 100
lr_plateau_factor = 0.1
epochs = 500
batch_size = 1024
validation_split = 0.2
'''
model storage and name
'''
model_name = "CodeMixed-Emb-BiLSTM_32x2_64x1_DO-Dense_64x2-{}".format(int(time.time()))


def save_model(model_qualifier, model, location):
    name = model_qualifier + "-{}".format(datetime.date.today()) + "-{}".format(time.time())
    try:
        model.save(os.path.join(location, name))
    except IOError:
        print("Exception occured while saving the model to disc.")


def set_keras_callbacks():
    es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=patience, restore_best_weights=True)
    mc = ModelCheckpoint(model_dir + 'Best' + model_name + '.h5', monitor='val_acc', mode='min', verbose=1)
    lrp = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=lr_plateau_factor, patience=patience,
                                                      verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    return es, mc, lrp


x_train, y_train, x_test, y_test, embedding_matrix = dataset.dataset()
K.clear_session()
model = model.model(max_words, embedding_dim, maxlen)
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model.summary()

optimizer = Adam(lr=learning_rate, decay=lr_decay)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

'''
Callbacks from Keras
'''
early_stopping, reduce_lr_plateau, model_checkpoint = set_keras_callbacks()
'''
Tensor board setup
'''
tensor_board = TensorBoard(log_dir=tensorBoard_logs_dir + '{}'.format(model_name), histogram_freq=1)

'''
Model training
'''
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split= validation_split,
                    callbacks=[tensor_board, early_stopping, reduce_lr_plateau, model_checkpoint])
save_model(model_name, model, ".\\models")

'''
Visualize the performance of the model
'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
run_epochs = range(1, len(acc)+1)

plt.plot(run_epochs, acc, 'bo', label='Training acc')
plt.plot(run_epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(run_epochs, loss, 'bo', label='Training Loss')
plt.plot(run_epochs, val_loss, 'b', label='Validation Loss')
plt.title("Training and validation Losses")
plt.legend()
plt.figure()
plt.show()