import os
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
import numpy as np

#print(len(os.listdir("./train/audio/bed")))

'''the audios files were saved in directories (word = dir).
    We will work only on the half of the data. Also we divide the data into train (80%) and test(20%)
    after separate the data we need to read the wavs file using scipy library then we apply MFCC.
    The MFCC of '''
def process_data(files_dir):
    audio_dir = os.listdir(files_dir)
    mfcc_list_train = []
    mfcc_vector_max = []
    mfcc_list_test = []
    for audio in audio_dir[:2]:
        print(audio)
        mfcc_list_data = read_files(files_dir + audio)
        mfcc_list_train.append(mfcc_list_data[0])
        mfcc_vector_max.append(mfcc_list_data[1])
        mfcc_list_test.append(mfcc_list_data[2])
    max_vector = max(mfcc_vector_max)
    label_index = 0
    train_data = []
    for mfcc_train in mfcc_list_train:
        train_data.extend(create_data(mfcc_train, max_vector, label_index))
        label_index += 1

    print(train_data)
    #return read_files(train_list,files_dir)
def read_wav(dir,files):
    signals_rates = []
    for file in files:
        rate, signal = wavfile.read(dir + "/" + file)   #read the file
        signals_rates.append([signal, rate])    #add the file to the list
    return signals_rates
def read_files(dir):
    #read wav files
    files = os.listdir(dir)
    files_leng = int(len(files) / 2)
    #print(len(files[:files_leng]))
    train_list = files[:round(files_leng * 0.8)]
    test_list = files[round(files_leng * 0.8) +1:files_leng]
    signals_rates_train = read_wav(dir, train_list)
    signals_rates_test = read_wav(dir, test_list)
    # extracting  feature(using mfcc) from wav files (sig_rate)
    # mfcc_list is a list of vector of size NUMFRAMES by numcep
    # max_mfcc_vector is the max of mfcc_list size of vectors
    mfcc_list_train, max_mfcc_vector = wav_to_mfcc(signals_rates_train)
    mfcc_list_test, max_test_vect = wav_to_mfcc(signals_rates_test)
    return mfcc_list_train,max_mfcc_vector,mfcc_list_test
# extracting  feature (using mfcc)
#return features and the max length vector(the longest word)
def wav_to_mfcc(sig_rate):
    mfcc_list = []  #initialize the mfcc list
    max_mfcc_vector= 0  #initialize the max
    j = 0
    for i in sig_rate:
        #apply mfcc for each wave file and save it in the mfccc_list
        mfcc_list.append(mfcc(i[0], i[1],winlen=0.025,winstep=0.009,nfilt=20,numcep=13))
        leng = len(mfcc_list[-1])
        #find the longest phoneme/word
        if(max_mfcc_vector<leng):
            k = j
            max_mfcc_vector  = leng
        j = j + 1
    return mfcc_list,max_mfcc_vector*len(mfcc_list[0][0])
#read wav failes and extract features from the signals
def create_data(mfcc_list,max_len,label_index):
    data = []
    for f in mfcc_list:
        feature = np.array(f).reshape(-1)
        feacture_len = len(feature)
        feature = np.append(feature, np.zeros((1, max_len - feacture_len)))
        # create label
        label = np.insert(np.zeros(29), label_index, 1)  # create one hot vector
        data.append([feature ,label])
    return data
process_data("./train/audio/")