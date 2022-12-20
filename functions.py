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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import librosa



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
    
    # WAVE_OUTPUT_FILENAME = "Record.wav"
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
    print(speakers)
    print(log_likelihood)
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
    return [access ,detected_model, log_likelihood , speakers  ] 

def create_MFCC_img(filename):
        # y, samplerate = librosa.load('record/Record.wav')
        y, samplerate = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=y, sr=samplerate, n_mfcc=10)
        figure=plt.figure()
        img = librosa.display.specshow(mfccs, x_axis='time')
        figure.colorbar(img)
        plt.savefig('static/Css/img/MFCC.png', bbox_inches='tight',pad_inches = 0)
        plt.tight_layout()
        plt.close()
        im = Image.open('static/Css/img/MFCC.png')
        data = io.BytesIO()
        im.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())
        return encoded_img_data


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=46)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_features[22]

def create_normal_img(filename):
    omar_f = features_extractor("normal_records\omar_open_door-sample0.wav")
    neven_f = features_extractor("normal_records\open_door_neveen-sample1.wav")
    salman_f = features_extractor("normal_records\open_the_door_nourhan-sample1.wav")
    nur_f = features_extractor("normal_records\open_the_door_salman-sample1.wav")
    other_f = features_extractor(filename)
    #define multiple normal distributions
    plt.plot(omar_f, norm.pdf(omar_f, np.mean(omar_f), np.std(omar_f)), label='Omar', color='violet')
    plt.plot(neven_f, norm.pdf(neven_f, np.mean(neven_f), np.std(neven_f)), label='Neven', color='red')
    plt.plot(salman_f, norm.pdf(salman_f, np.mean(salman_f), np.std(salman_f)), label='Salman', color='grey')
    plt.plot(nur_f, norm.pdf(nur_f, np.mean(nur_f), np.std(nur_f)), label='Nur', color='blue')
    plt.plot(other_f, norm.pdf(other_f, np.mean(other_f), np.std(other_f)), label='Test', color='black')
    # plt.plot(other_f, norm.pdf(other_f, np.mean(other_f), np.std(other_f)), label='Other', color='black')
    #add legend to plot
    plt.legend(title='Parameters')
    #add axes labels and a title
    plt.ylabel('Density')
    plt.xlabel('MFCC')
    plt.title('Normal Distributions', fontsize=14)
    plt.savefig('static/Css/img/Normal.png', bbox_inches='tight',pad_inches = 0)
    plt.tight_layout()
    plt.close()
    im = Image.open('static/Css/img/Normal.png')
    data = io.BytesIO()
    im.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())



def create_scores_img(log_likelihood , speakers):
    data = {'Omar':-log_likelihood[2], 'Neven':-log_likelihood[0], 'Nurhan':-log_likelihood[1],
        'Salman':-log_likelihood[5]}
    scores = list(data.keys())
    team = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(scores, team, color ='maroon',width = 0.4)
    plt.axhline(y=24.5,linewidth=3,color='b')

    # plt.xlabel("Courses offered")
    plt.ylabel("Likelihood")
    plt.title("GMM Match")
    # plt.show()

    plt.savefig('static/Css/img/scores.png', bbox_inches='tight',pad_inches = 0)
    plt.tight_layout()
    plt.close()
    im = Image.open('static/Css/img/scores.png')
    data = io.BytesIO()
    im.save(data, "PNG")