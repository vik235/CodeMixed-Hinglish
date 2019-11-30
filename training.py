import os
import keras
import io
import csv
import time
import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Dropout, SimpleRNN, LSTM, Bidirectional
from keras.layers import BatchNormalization
from keras.optimizers import Adam
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

'''
Models will be saved in the models folder in the root. We expect that to be present and will be 
available if the repo is initialized
model_qualifier: The friendly name via which the model needs to saved: for example 
Best_Hinglish-Emb-TFL-LSTM_32x4_DO-Dense_64x2_model means that the models is among best models for Hinglish classifier
and has en learnt embedding with LSTM layers with activation units 32 stacked into 4 and has a recurrent drop out 
enabled follwoed by 2 FC layers of 64 neurons each.   
'''


def save_model(model_qualifier, model, location):
    model_name = model_qualifier + "-{}".format(datetime.date.today()) + "-{}".format(time.time())
    try:
        model.save(os.path.join(location, model_name))
    except IOError:
        print("Exception occured while saving the model to disc.")


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
print('Found %s unique tokens.'% len(word_index))

'''
padding the sequences to make them all wequal
'''
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of the data tensor:' , data.shape)
print('Shape of the label tensor:' , labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices, :]

x_train = data[:training_samples]
y_train = labels[:training_samples]
print(indices[0:10])
x_test = data[training_samples:training_samples + testing_samples]
y_test = labels[training_samples:training_samples + testing_samples]

print("Train length" , len(x_train))
print("Test length" , len(x_test))


'''
Loading the embeddings from Glove
'''
embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
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

np.random.seed(1)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
'''
model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
model.add(LSTM(32))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(Dropout(0.6))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(Dropout(0.7))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(Dropout(0.6))
model.add(GlobalMaxPooling1D())
'''
model.add(Bidirectional(LSTM(32, recurrent_dropout = 0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(32, recurrent_dropout = 0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(32, recurrent_dropout = 0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
#model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(3, activation = 'softmax'))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model_name = "CodeMixed-Emb-BiLSTM_32x2_64x1_DO-Dense_64x2-{}".format(int(time.time()))
model.summary()

optimizer = Adam(learning_rate=0.01, decay=1e-6)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['acc'])

#callbacks
earlystopping = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=40, restore_best_weights= True)

modelcheckpoint = ModelCheckpoint('.\\models\Best_Hinglish-Emb-BiLSTM_32x2_64x1_DO-Dense_64x2_model.h5', monitor='val_acc', mode='min', verbose=1)

reduce_lr_plateau = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)


tensorboard = TensorBoard(log_dir = tensorBoard_logs_dir + '{}'.format(model_name),
                         histogram_freq=1,
                         #embeddings_freq=1
                         )

'''
TensorBoard(batch_size=batch_size,
                          embeddings_freq=1,
                          embeddings_layer_names=['features'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=x_test)

'''

history = model.fit(x_train, y_train, epochs = 500, batch_size = 1024,validation_split=0.1, callbacks = [tensorboard, earlystopping, reduce_lr_plateau, modelcheckpoint])

save_model("CodeMixed-Emb-BiLSTM_32x2_64x1_DO-Dense_64x1_model", model, ".\\models")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label ='Training Loss')
plt.plot(epochs, val_loss, 'b', label ='Validation Loss')
plt.title("Training and validation Losses")
plt.legend()
plt.figure()

plt.show()