#import  libraries---------------------------------------------------------
from flask import Flask , render_template
import pickle
import numpy as np
import os
import wave
import pyaudio
import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
import librosa
import librosa.display
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from werkzeug.utils import secure_filename



def record_audio_test():
    
    now = datetime.now() 
    filename =now.strftime("%m/%d/%Y, %H:%M:%S")+".wav"
    filename=secure_filename(filename)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 2
    audio = pyaudio.PyAudio()
    index = 1	
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = index,
                frames_per_buffer=CHUNK)
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # OUTPUT_FILENAME="Record.wav"
    

    WAVE_OUTPUT_FILENAME=os.path.join("record",filename)
    trainedfilelist = open("record.txt", 'a')
    trainedfilelist.write(filename+"\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    return WAVE_OUTPUT_FILENAME

def calculate_delta(array):
    rows,cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def test_model(model_name,modelpath,filename):
    gmm_files = [os.path.join(modelpath,fname) for fname in
                os.listdir(modelpath) if fname.endswith('.gmm')]
    #Load the Gaussian mixture Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
    # path = "record\Record.wav"
    path=filename
    print(f"\n Test File : {path} ")
    sr,audio = read(path)
    vector   = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models)); 
    max_score=-100
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        if scores > max_score:
            max_score = scores
        print(gmm_files[i])
        print(scores)
        log_likelihood[i] = scores.sum()
    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    detected_model=speakers[winner]
    print(max_score)
    print("-"*50)
    access = False
    if (model_name == "people"):
        if max_score > -24.5:
            access = True
            print("In Group")
        else :
            access = False
            print("Other")
    print("#"*50)
    time.sleep(1.0)  
    return access ,detected_model 
