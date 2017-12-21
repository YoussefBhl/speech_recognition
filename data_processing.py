import os
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
#print(len(os.listdir("./train/audio/bed")))

'''process_data takes the files directory
    we divide the data in half then we split data into train and test
    finally we aplly MFCC and save it in pickle file '''
def process_data(files_dir):
    files = os.listdir(files_dir)
    files_leng = int(len(files)/2)
    print(len(files[:files_leng]))
    train_list = files[:round(files_leng*0.8)]
    test_list = files[round(files_leng*0.8)+1:]


# extracting  feature (using mfcc)
#return features and the max length vector(the longest phonme/word)
def wav_to_mfcc(sig_rate):
    mfcc_list = []  #initialize the mfcc list
    max_mfcc_vector= 0  #initialize the max
    j = 0
    for i in sig_rate:
        #apply mfcc for each wave file and save it in the mfccc_list
        mfcc_list.append(mfcc(i[0], i[1],winlen=0.025,winstep=0.009,nfilt=20,numcep=13))
        #nfilt=20   0.730
        #ceplifter=0  0.78
        leng = len(mfcc_list[-1])
        #find the longest phoneme/word
        if(max_mfcc_vector<leng):
            k = j
            max_mfcc_vector  = leng
        j = j + 1
    return mfcc_list,max_mfcc_vector*len(mfcc_list[0][0]),k
#read wav failes and extract features from the signals
def read_files(files,dir):
    #read wav files
    signals_rates = []  #initialize the signals_rate list
    for file in files:
        rate, signal = wavfile.read(dir + "/" + file)   #read the file
        signals_rates.append([signal, rate])    #add the file to the list
    # extracting  feature(using mfcc) from wav files (sig_rate)
    # mfcc_list is a list of vector of size NUMFRAMES by numcep
    # max_mfcc_vector is the max of mfcc_list size of vectors
    mfcc_list, max_mfcc_vector,j = wav_to_mfcc(signals_rates)
    return mfcc_list,max_mfcc_vector
process_data("./train/audio/bed")