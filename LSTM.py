import librosa
import numpy as np
from IPython.lib.display import Audio
import tensorflow as tf
import tflearn
import pandas as pd
from random import shuffle
import os


learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10 # digits


def mfcc_batch_generator(batch_size=20):
    # maybe_download(source, DATA_DIR)
    # if target == Target.speaker: speakers = get_speakers()
    batch_features = []
    labels = []
    files = os.listdir("./train/audio/yes")
    while True:
        # print("loaded batch of %d files" % len(files))
        # shuffle(filename)
        for file in files:
            # if not file.endswith(".wav"): continue
            wave, sr = librosa.load("./train/audio/yes" + "/" + file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            # if target==Target.speaker: label=one_hot_from_item(speaker(file), speakers)
            # elif target==Target.digits:  label=dense_to_one_hot(int(file[0]),10)
            # elif target==Target.first_letter:  label=dense_to_one_hot((ord(file[0]) - 48) % 32,32)
            # elif target == Target.hotword: label = one_hot_word(file, pad_to=max_word_length)  #
            # elif target == Target.word: label=string_to_int_word(file, pad_to=max_word_length)
            # label = file  # sparse_labels(file, pad_to=20)  # max_output_length
            # else: raise Exception("todo : labels for Target!")
            label = np.insert(np.zeros(9), 0, 1)
            labels.append(label)
            print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
            print(np.array(mfcc).shape)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                # if target == Target.word:  labels = sparse_labels(labels)
                # labels=np.array(labels)
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                # print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)
                yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []

batch = word_batch = mfcc_batch_generator(64)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y


# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, x )


model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)