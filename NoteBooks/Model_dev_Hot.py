#!/usr/bin/env python
# coding: utf-8

# In[94]:


import os
import keras
import io
import csv

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
import numpy as np
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
import keras_metrics
from keras.utils.vis_utils import plot_model


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D 

from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot

## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.manifold import TSNE
import time


# In[2]:


##Set params for the model
tensorBoard_logs_dir = 'C:\\Users\\vigupta\\OneDrive\\Learning\\DataScience\\DeepLearning\\CS230\\Project\\Hinglish-Offensive-Text-Classification-master\\logs\\'
hottransmsg_dir = 'C:\\Users\\vigupta\\OneDrive\\Learning\\DataScience\\DeepLearning\\CS230\\Project\\Hinglish-Offensive-Text-Classification-master'
model_dir = 'C:\\Users\\vigupta\\OneDrive\\Learning\\DataScience\\DeepLearning\\CS230\\Project\\Hinglish-Offensive-Text-Classification-master\\models\\'
glove_dir = 'C:\\Users\\vigupta\\OneDrive\\Learning\\DataScience\\DeepLearning\\Glove'
train_dir = os.path.join(hottransmsg_dir, 'train')
train_data = 'transmessages.csv'
label_data = 'labels.csv'
maxlen = 100
training_samples = 2500
validation_samples = 500# len(labels) - training_samples #
max_words = 10000
num_filters = 64 
embedding_dim = 100
num_filters = 64 


# In[29]:


texts = []
labels = []

with open(os.path.join(train_dir, train_data), newline ='') as messageData:
    reader = csv.reader(messageData)
    for row in reader:
        message = (''.join(row))
        texts.append(message)

with open(os.path.join(train_dir, label_data)) as labelData:
    reader = csv.reader(labelData)
    for row in reader:
        #label = (''.join(row))
        labels.append(row)
print(type(labels))

labels = [(np.squeeze(i)) for i in labels]
labels


# In[32]:


'''
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels,)
labells = np_utils.to_categorical(encoded_Y)
print(labells[0,])

'''
labells = np_utils.to_categorical(labels)


# In[5]:


'''
This code block is to be ignored; borrowed for a different project
try:
    for label_type in ['neg' , 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding="utf8")
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else :
                    labels.append(1)
except:
    print("Error happened in reading and processing the dataset")
print('The length of the dataset is:', len(labels))            
print('The length of the texts is:', len(texts))            
'''


# In[105]:


##Tokenizer, Sequencer and padding via Keras
np.random.seed(1) 

#tokenize the data for the maxwords
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

#Generate the sequences on texts on the the data by tokenizer
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.'% len(word_index))
#print(word_index)
#Padding sequences (making them all equal)
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of the data tensor:' , data.shape)
print('Shape of the label tensor:' , labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labells[indices,:]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_test = data[training_samples:training_samples + validation_samples]
y_test = labels[training_samples:training_samples + validation_samples]

print("Train length" , len(x_train))
print("Validation length" , len(x_val))

with open(os.path.join(tensorBoard_logs_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, y_test)


# In[7]:


y_val


# In[35]:


np.random.seed(1) 
embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
   # print(word)
    coefs = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coefs
f.close()

print('Found %s word vectors' % len(embedding_index))


# In[36]:


np.random.seed(1) 
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)       
#embedding_matrix[0:10, ]


# In[89]:


np.random.seed(1) 
model = Sequential() 
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
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
model.summary()
model_name = "CodeMixed-HOT-Emb-TFF-CONV-64x3x7-MP-Dense-64x2-BN-DO-{}".format(int(time.time()))


# In[90]:


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])


# In[91]:



tensorboard = TensorBoard(log_dir = tensorBoard_logs_dir + '{}'.format(model_name))
'''
TensorBoard(batch_size=batch_size,
                          embeddings_freq=1,
                          embeddings_layer_names=['features'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=x_test)

'''

history = model.fit(x_train, y_train, epochs = 40, batch_size = 256,validation_split=0.1, callbacks = [tensorboard])
results = model.evaluate(x_test, y_test, callbacks = [tensorboard] )
model.save_weights(model_dir + '{}'.format(model_name)+'.h5')


# In[81]:


os.getcwd()


# In[92]:


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


# In[85]:


print(model.metrics_names)
results


# In[100]:


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

weights[1:20, 0:10]


# In[76]:


## Plotting function
def plot_words(data, start, stop, step):
    trace = go.Scatter(
        x = data[start:stop:step,0], 
        y = data[start:stop:step, 1],
        mode = 'markers',
        text= word_list[start:stop:step]
    )
    layout = dict(title= 't-SNE 1 vs t-SNE 2',
                  yaxis = dict(title='t-SNE 2'),
                  xaxis = dict(title='t-SNE 1'),
                  hovermode= 'closest')
    fig = dict(data = [trace], layout= layout)
    py.iplot(fig)


# In[51]:


tsne_embeddings = TSNE(n_components=2).fit_transform(weights)
plot_words(tsne_embeddings, 0, 2, 1)


# In[ ]:




