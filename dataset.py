
import os
import csv

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


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
np.random.seed(1)


def prepare_data(training_directory, training_data, labelled_data):
    texts = []
    labels = []
    try:
        with open(os.path.join(training_directory, training_data), newline='') as messageData:
            reader = csv.reader(messageData)
            for row in reader:
                message = (''.join(row))
                texts.append(message)
    except IOError:
        print("Error reading the texts files")

    try:
        with open(os.path.join(training_directory, labelled_data)) as labelData:
            reader = csv.reader(labelData)
            for row in reader:
                labels.append(row)
            labels = np.squeeze(labels)
    except IOError:
        print("Error reading the labels  files")
    return texts, labels


def dataset():
    texts, labels = prepare_data(train_dir, train_data, label_data)
    labels = np_utils.to_categorical(labels, num_classes=num_classes)

    '''
    #tokenize the data for the maxwords
    '''

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    '''
    #Generate the sequences on texts on the the data by tokenizer
    '''

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    '''
    padding the sequences to make them all equal
    '''
    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of the data tensor:', data.shape)
    print('Shape of the label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices, :]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    print(indices[0:10])
    x_test = data[training_samples:training_samples + testing_samples]
    y_test = labels[training_samples:training_samples + testing_samples]

    print("Train length", len(x_train))
    print("Test length", len(x_test))

    '''
    Loading the embeddings from Glove
    '''
    embedding_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors' % len(embedding_index))

    np.random.seed(1)
    embedding_matrix = np.zeros((max_words, embedding_dim))
    count = 0
    ignored_words = []
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                ignored_words.append(word)
                count = count + 1
    print(embedding_matrix.shape)
    return x_train, y_train, x_test, y_test, embedding_matrix



