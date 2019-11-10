#!/usr/bin/env python
# coding: utf-8

# In[4]:


import re
import io
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from googletrans import Translator
import unicodedata


# In[5]:


def StandardizeCasing(message):
    return message.lower()


# In[6]:


def RemoveUsermentions(message):
    regex = re.compile('@[A-Za-z0-9_-]{2,}', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    message = re.sub(regex, '', message)
    return message.strip()
    


# In[7]:


def RemoveUrls(message):
    regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.S)
    message = re.sub(regex, '', message)
    return message.strip()


# In[8]:


def RemoveEmoji(message):
    #regex = re.compile('\[a-z]*',re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    regex = re.compile(r'\\x[A-Za-z0-9]*', re.S)
    message = re.sub(regex, '', message)
    return message.strip()
    #return message.encode('ascii', 'ignore').decode('ascii')
    


# In[9]:


def RemovePunctuation(message):
    regex = re.compile('[,:/\"[\]]', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    message = re.sub(regex, '', message)
    return message.strip()
    


# In[10]:


def RemoveDigits(message):
    regex = re.compile('[0-9]+', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    message = re.sub(regex, '', message)
    return message.strip()
    


# In[11]:


def RemoveUnicode(message):
    return unicodedata.normalize('NFKD', message).encode('ascii','ignore')
    


# In[12]:


def RemoveNewlinewithin(message):
    #regex = re.compile('\x', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    #message = re.sub(regex, '', message)
    #return message.strip()
    return message.replace('\\n', ' ')
    


# In[13]:


def removeDots(message):
    regex = re.compile('[.]+', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    message = re.sub(regex, ' ', message)
    return message.strip()


# In[14]:


def RemoveSpecialChars(message):
    #regex = re.compile('[?*\"\'!#$^\&~`@]', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    regex = re.compile(r"[-()\"#/@;:<>{}`+=~|.!?,]", re.S)
    message = re.sub(regex, '', message)
    return message.strip()


# In[15]:


def RemoveRT(message):
    regex = re.compile('rt ', re.S) #remove the @ sign with any trailing letters, digits or _ 2 or more
    message = re.sub(regex, '', message)
    return message.strip()


# In[16]:


def RemoveDEmoji(message):
    emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', message)


# In[77]:


def PreprocessMessage(message):
    message = RemoveEmoji(message)
    message = StandardizeCasing(message)
    message = RemoveUsermentions(message)
    message = RemoveUrls(message)
    message = RemovePunctuation(message)
    message = RemoveDigits(message)
    message = RemoveNewlinewithin(message)
    message = removeDots(message)
    message = RemoveSpecialChars(message)
    message = RemoveRT(message)
    return message


# In[17]:


message = "haa jaise tum bhi abhi p\xe2\x80\xa6"
print(RemoveUnicode(message))


# In[81]:


messages = []
with open('messages_cleaned.csv', newline ='') as messageData:
    reader = csv.reader(messageData)
    for row in reader:
        message = (''.join(row))
        message = PreprocessMessage(message)
        messages.append(message)
        
    
        


# In[82]:


with open('messages.csv', mode='w') as tweets:
    writer = csv.writer(tweets, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([messages])


# In[85]:


messages_trans = []
with open('messages.csv', newline ='') as messageData:
    reader = csv.reader(messageData)
    translator = Translator()
    for message in reader:
        trans = translator.translate(message, src = 'hi', dest = 'en')
        print(trans.text)
        
        
        
    
        


# In[ ]:




