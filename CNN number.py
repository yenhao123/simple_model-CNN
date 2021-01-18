#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
np.random.seed(10)


# In[2]:


# beforehand
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
## convert to normalization
x_train_nor = x_train / 255
x_test_nor = x_test /255
## convert to one_hot
y_train_one = np_utils.to_categorical(y_train)
y_test_one = np_utils.to_categorical(y_test)


# In[3]:


# model setting
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))


# In[4]:


# loss function setting
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[5]:


# start training
train_history = model.fit(x=x_train_nor,y=y_train_one,validation_split=0.1,epochs=15,batch_size=300,verbose=2)


# In[44]:


def plot_images_labels(images,labels,prediction,idx,num=5):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    ## predict
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(np.reshape(images[idx+i],(28,28)),cmap='binary')
        title = "label:" + str(labels[idx+i]) + "pedict:" + str(prediction[idx+i])
        ax.set_title(title,fontsize=10)
    plt.show()


# In[45]:


## prediction
prediction = model.predict_classes(x_test_nor)
plot_images_labels(x_test,y_test,prediction,idx=300)

