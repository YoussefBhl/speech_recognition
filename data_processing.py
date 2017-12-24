import os
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
import numpy as np
import pickle
from random import shuffle

'''the audios files were saved in directories (word = dir).
    we divide the data into train (80%) and test(20%)
    after separate the data we need to read the wavs file using scipy library then we apply MFCC.
     finally we save it in a pickle file'''
def process_data(files_dir):
    audio_dirs = os.listdir(files_dir)#all the direcotries that contains data
    mfcc_list_train = []
    mfcc_vector_max = []
    mfcc_list_test = []
    for dir in audio_dirs:
        print("processing : " + dir)
        #we readd all the data (wav) then return the extracted mfcc for train and test
        #with the maximum of vector for each word (we will need it to pad the data )
        mfcc_list_data = read_files(files_dir + dir)
        mfcc_list_train.append(mfcc_list_data[0])
        mfcc_vector_max.append(mfcc_list_data[1])
        mfcc_list_test.append(mfcc_list_data[2])
    max_vector = max(mfcc_vector_max)
    label_index = 0
    train_data = []
    #we save the data in pickle file
    pickle_data(mfcc_list_train, "train.p", max_vector)
    pickle_data(mfcc_list_test, "test.p", max_vector)
    '''for mfcc_train in mfcc_list_train:
        train_data.extend(create_data(mfcc_train, max_vector, label_index))
        label_index += 1'''

#create data and save it in pickle file
def pickle_data(data,pickle_file,max_vector):
    pickle_ = []
    label_index = 0
    #we pad the extracted mfcc (wich is the representation of the wav file) and create it label
    for mfcc in data:
        pickle_.extend(create_data(mfcc, max_vector, label_index))
        label_index += 1
    pickle.dump(pickle_, open(pickle_file, "wb"))


def read_wav(dir,files):
    signals_rates = []
    for file in files:
        rate, signal = wavfile.read(dir + "/" + file)   #read the file
        signals_rates.append([signal, rate])    #add the file to the list
    return signals_rates
def read_files(dir):
    #read wav files
    files = os.listdir(dir)
    #we need to shuffle the data cuz we'll devide the data
    shuffle(files)
    files_leng = int(len(files))
    train_list = files[:round(files_leng * 0.8)]
    test_list = files[round(files_leng * 0.8) +1:files_leng]
    #we read the files (wav) wich return the list of signals and rate for each file
    signals_rates_train = read_wav(dir, train_list)
    signals_rates_test = read_wav(dir, test_list)
    # extracting  feature(using mfcc) from wav files (sig_rate)
    # mfcc_list is a list of vector of size NUMFRAMES by numcep
    # max_mfcc_vector is the max of mfcc_list size of vectors
    mfcc_list_train, max_mfcc_vector = wav_to_mfcc(signals_rates_train)
    mfcc_list_test, max_test_vect = wav_to_mfcc(signals_rates_test)
    return mfcc_list_train,max_mfcc_vector,mfcc_list_test
# extracting  feature (using mfcc)
#return features and the max length vector(we will use it later the pad the data)
def wav_to_mfcc(sig_rate):
    mfcc_list = []  #initialize the mfcc list
    max_mfcc_vector= 0  #initialize the max
    for i in sig_rate:
        #apply mfcc for each wave file and save it in the mfccc_list
        mfcc_list.append(mfcc(i[0], i[1],winlen=0.025,winstep=0.009,nfilt=20,numcep=13))
        leng = len(mfcc_list[-1])
        #find the longest phoneme/word
        if(max_mfcc_vector<leng):
            max_mfcc_vector  = leng
        j = j + 1
    return mfcc_list,max_mfcc_vector
#read wav failes and extract features from the signals
def create_data(mfcc_list,max_len,label_index):
    data = []
    for f in mfcc_list:
        feacture_len = len(f)
        # create label (we have 30 classes)
        label = np.insert(np.zeros(29), label_index, 1)  # create one hot encoding
        #padding the features (all the data should have the same shape)
        feature = np.pad(f, ((0, max_len - feacture_len), (0, 0)), mode='constant', constant_values=0)
        data.append([feature ,label])
    # return the list of features and labels
    return data
process_data("./train/audio/")