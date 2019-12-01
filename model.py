from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D

from keras.layers import Dropout, SimpleRNN, LSTM, Bidirectional
from keras.layers import BatchNormalization

maxlen = 200              # Maximum length of the message/ tweet to be considered
training_samples = 2500
testing_samples = 3189 - training_samples
max_words = 10000         # vocab length
num_filters = 64
num_classes = 3
num_hidden_lstm = 32
num_hidden_rnn_final = 32
num_dense_fc = 64
recurrent_dropout = 0.2
fc_dropout = 0.5
cnn_dropout = 0.5
kernel_size = 7


def model(vocab_size, embedding_dimension, input_length):
    max_words = vocab_size
    embedding_dim = embedding_dimension
    maxlen = input_length
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen), name='Embedding_Layer-Hinglish ')
    '''
    model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
    model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
    model.add(LSTM(32, recurrent_dropout = 0.2, return_sequences=True))
    model.add(LSTM(32))

    model.add(Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(Dropout(cnn_dropout))
    model.add(GlobalMaxPooling1D())
    '''
    model.add(Bidirectional(LSTM(num_hidden_lstm, recurrent_dropout=recurrent_dropout, return_sequences=True)), name='BiLSTM-32-0.2-Seq-01')
    model.add(Bidirectional(LSTM(num_hidden_lstm, recurrent_dropout=recurrent_dropout, return_sequences=True)), name='BiLSTM-32-0.2-Seq-02')
    model.add(Bidirectional(LSTM(num_hidden_lstm, recurrent_dropout=recurrent_dropout, return_sequences=True)), name='BiLSTM-32-0.2-Seq-03')
    model.add(Bidirectional(LSTM(num_hidden_lstm)), name='BiLSTM-32-NoSeq-04')
    model.add(Dense(num_dense_fc, activation='relu'), name='Dense-64-Relu')
    model.add(BatchNormalization(), name='BN')
    model.add(Dropout(fc_dropout), name='DropOut-0.5')
    model.add(Dense(num_dense_fc, activation='relu'), name='Dense-64-Relu')
    model.add(Dropout(fc_dropout), name='DropOut-0.5')
    model.add(BatchNormalization(), name='BN')
    model.add(Dense(num_classes, activation='softmax'), name='Dense-3-SoftMax')
    print(model.summary())
    return model


