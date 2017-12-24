import numpy as np
import tensorflow as tf
import tflearn
from random import shuffle
import os
import pickle

learning_rate = 0.001
training_iters = 1  # steps
batch_size = 1000
width = 110  # mfcc features
height = 13  # (max) length of utterance
classes = 30 # digits

#the labls and the features are saved in the data
#we need to split the data into X (features) and Y (labels)
def separate_data(data):
    x = []
    y = []
    for index, d in enumerate(data):
        x.append(np.array(d[0]))
        y.append(np.array(d[1]))
    return x, y

#load the date from the pickles files (trai and test)
def load_data():
    train = pickle.load(open("train.p", "rb"))
    test = pickle.load(open("test.p", "rb"))
    return train,test

#load the data and split it
train, test = load_data()
testX, testY = separate_data(test)
trainX,trainY = separate_data(train)
# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 200, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, x )
model = tflearn.DNN(net, tensorboard_dir='./tflearn_logs/', tensorboard_verbose=0)
#fit the model
model.fit(trainX, trainY,n_epoch=20, validation_set=(testX, testY),
          shuffle =True, show_metric=True,batch_size=batch_size)
_y=model.predict(trainX)
model.save("tflearn.lstm.model")
