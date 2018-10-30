
# coding: utf-8

# In[106]:


'''
Importing Modules to be used in the program
'''
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import gensim
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import math
import numpy as np
import math
import random
from operator import itemgetter


# In[107]:


'''
Using nltk for stop word and Lemmatizing
'''
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
import pandas as pd


# In[108]:


'''
Reading Training File
'''
with open('full_train.txt') as f:
    content = f.readlines()


# In[109]:


'''
Preprocessing data for spliting labels and documents apart and forming a database forfurther processing
'''

documents = []
label_list = []
lis_len = len(content)
count = 0
dictionary = []
for line in content:
    count +=1
    if count%10000 == 0:
        print("Percentage data preprocessed", (count/lis_len)*100)
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    text = sentence.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text if ((word not in stopwords) and len(word)>2)])
    documents.append(text)
    temp_list = []
    for label in labels.split(','):
        label = label.strip()
        temp_list.append(label)
        dictionary.append(label)
    label_list.append(temp_list)


# In[110]:


'''
Importing TF-idf vectorizer for feature computation
'''

vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(documents)


# In[111]:


'''
Creating Dictionary for translation b/w labels and the corresponding indexes
'''
uniques = np.unique(dictionary)
lab_2_index = {}
index_2_lab = {}
for i, j in enumerate(uniques):
    lab_2_index[j] = i
    index_2_lab[i] = j


# In[112]:


label_list_num = []
for item in label_list:
    temp_list = []
    for label in item:
        temp_list.append(lab_2_index[label])
    label_list_num.append(temp_list)


# In[113]:


'''
Forming a complete dataset with dataset and corresponding labels in main_data and main_label
'''

main_data = []
main_label = []
for doc, label in zip(X, label_list):
    for lab in label:
        main_data.append(doc.toarray())
        main_label.append(lab_2_index[lab])
main_data = np.array(main_data).reshape(np.array(main_data).shape[0], np.array(main_data).shape[2])
main_label = np.array(main_label)


# In[114]:


#temp = np.zeros((len(main_label), 50))
#temp[np.arange(len(main_label)), main_label] =1
#main_label = temp


# In[115]:


with open('items_train.pkl', 'wb') as f:
    pickle.dump([main_data, main_label, index_2_lab, label_list_num, X], f, protocol=1)


# In[116]:


'''
Function for training the model, returns the weight matrix for all the classes.
'''
def train_model(X, y, epochs, reg_strength, batch_size, learning_rate, learning_mode, decay, write):
    lossy = []
    n_features = X.shape[1]
    n_classes = y.max() + 1
    W = np.random.randn(n_features, n_classes) / np.sqrt(n_features/2)
    config = {'reg_strength': reg_strength, 
                'batch_size': batch_size,
                'decay': decay,
                'learning_rate': learning_rate,
                'learning_mode' : learning_mode,
                'eps': 1e-8}
    for curr_epoch in range(epochs):
        loss, config, W = SGD(X, y,W, config, curr_epoch)
        if curr_epoch%1000 ==0:
            if write :
                print ("Epochs Run: %s, Current Loss: %s" % (curr_epoch, loss))
            lossy.append(loss)
    return W, lossy


# In[117]:


'''
Computes the loss incurred during computing the model.
'''

def compute_loss(X, y, W, b, reg_strength):
    sample_size = X.shape[0]
    predictions = X.dot(W) + b
    predictions -= predictions.max(axis=1).reshape([-1, 1])
    softmax = math.e**predictions
    softmax /= softmax.sum(axis=1).reshape([-1, 1])
    loss = -np.log(softmax[np.arange(len(softmax)), y]).sum() 
    loss /= sample_size
    loss += 0.5 * reg_strength * (W**2).sum()

    softmax[np.arange(len(softmax)), y] -= 1
    dW = (X.T.dot(softmax) / sample_size) + (reg_strength * W)
    return loss, dW


# In[118]:


'''
Run a Stochastic Gradient Descent modelto train the Logistic Regression.
'''

def SGD(X, y, W, config,curr_epoch):
    items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'learning_mode', 'decay')(config)
    learning_rate, batch_size, reg_strength, learning_mode, decay = items

    loss, dW = randomize(X, y, batch_size, W, 0, reg_strength)
    if learning_mode == 'constant':
        W -= learning_rate * dW
    elif learning_mode == 'increasing':
        learning_rate *= (1. + decay * curr_epoch)
        W -= learning_rate * dW  
    elif learning_mode == 'decreasing':
        learning_rate *= (1. / (1. + decay * curr_epoch))
        W -= learning_rate * dW
    return loss, config, W


# In[119]:


'''
Randomizes data and Creates batches for computation.
'''
def randomize(X, y, batch_size, w, b, reg_strength):
    random_indices = random.sample(range(X.shape[0]), batch_size)
    X_batch = X[random_indices]
    y_batch = y[random_indices]
    return compute_loss(X_batch, y_batch, w, b, reg_strength)


# In[120]:


'''
Computes weight matrix by training the model and is used in subsequent calculations.
'''

weight ,loss  = train_model(np.array(main_data), np.array(main_label), 
                reg_strength= 1e-6, batch_size= 1000, 
                epochs= 10000, learning_rate= 1e-1 ,
                learning_mode = 'constant', decay= 1e-8, write = True)


# In[121]:


'''
Computes the model performance on the training set.
'''
prediction = np.argmax(np.array(X.toarray()).dot(weight), 1)
sum1 = 0
for i, j in zip(prediction ,label_list):
    if index_2_lab[i] in j:
        sum1 +=1
print('Training Accuracy of the model : ', sum1/len(label_list)*100)


# In[122]:


'''
Reading Test File
'''
with open('full_test.txt') as f:
    content = f.readlines()


# In[123]:


'''
Preprocessing data for spliting labels and documents apart and forming a database forfurther processing
'''

documents = []
label_list = []
lis_len = len(content)
count = 0
dictionary = []
for line in content:
    count +=1
    #if count%10000 == 0:
        #print("Percentage data preprocessed", (count/lis_len)*100)
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    text = sentence.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text if ((word not in stopwords) and len(word)>2)])
    documents.append(text)
    temp_list = []
    for label in labels.split(','):
        label = label.strip()
        temp_list.append(label)
        dictionary.append(label)
    label_list.append(temp_list)


# In[124]:


'''
Forming a complete dataset with dataset and corresponding labels in main_data and main_label
'''
X1 = vectorizer.transform(documents)
main_data1 = []
main_label1 = []
for doc, label in zip(X1, label_list):
    for lab in label:
        main_data1.append(doc.toarray())
        main_label1.append(lab_2_index[lab])
main_data1 = np.array(main_data1).reshape(np.array(main_data1).shape[0], np.array(main_data1).shape[2])
main_label1 = np.array(main_label1)


# In[125]:


label_list_num = []
for item in label_list:
    temp_list = []
    for label in item:
        temp_list.append(lab_2_index[label])
    label_list_num.append(temp_list)


# In[126]:


with open('items_test.pkl', 'wb') as f:
    pickle.dump([main_data1, main_label1, label_list_num, X1], f, protocol=1)


# In[128]:


'''
Computes the model performance on the training set.
'''
prediction = np.argmax(np.array(X1.toarray()).dot(weight), 1)
sum1 = 0
for i, j in zip(prediction ,label_list):
    if index_2_lab[i] in j:
        sum1 +=1
print('Test Accuracy of the model : ', sum1/len(label_list)*100)


# In[28]:


'''
For plot visualization.
'''
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = [10,8]
plt.style.use('fivethirtyeight')


# In[33]:


'''
Comparing performance for incresing, decreasing and constant learning rates.
'''
'''
1. Constant learning rate of 1e-1
'''
_, constant_loss = weight ,loss  = train_model(np.array(main_data), np.array(main_label), 
                    reg_strength= 1e-6, batch_size= 1000, 
                    epochs= 50000, learning_rate= 1e-1 ,
                    learning_mode = 'constant', decay= 1e-8, write = False)

'''
1. Decreasing learning rate with decay =  1e-8
'''
_, constant_loss_dec = weight ,loss  = train_model(np.array(main_data), np.array(main_label), 
                    reg_strength= 1e-6, batch_size= 1000, 
                    epochs= 50000, learning_rate= 1e-1 ,
                    learning_mode = 'decreasing', decay= 1e-8, write = False)
'''
1. Increasing learning rate with factor =  1e-8
'''
_, constant_loss_inc = weight ,loss  = train_model(np.array(main_data), np.array(main_label), 
                    reg_strength= 1e-6, batch_size= 1000, 
                    epochs= 50000, learning_rate= 1e-1 ,
                    learning_mode = 'increasing', decay= 1e-8, write = False)


# In[34]:


plt.plot(constant_loss, label = 'Constant Learning Rate')
plt.plot(constant_loss_dec, label = 'Decreasing Learning Rate')
plt.plot(constant_loss_inc, label = 'Increasing Learning Rate')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Variation with epoch for different Learning Schemes')
plt.show()

