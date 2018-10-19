
# coding: utf-8

# ## DNN on MNIST

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ### visulize an example

# In[5]:


eg=x_train[5].transpose()

get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(0,28):
    for j in range(0, 28):
        if eg[j,i]!=0:
            plt.scatter(j,i,color='g',marker='o')
plt.xlim([-10,60])
plt.ylim([-10,60])


# ### build a simple DNN

# In[7]:


batch_size = 50
num_classes = 10
epochs = 5

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train[:10000], y_train[:10000],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

