#import  libraries---------------------------------------------------------
from flask import Flask , render_template
import pickle
import numpy as np
import os
import wave
import pyaudio


def record_audio_test():
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
	OUTPUT_FILENAME="Record.wav"
	WAVE_OUTPUT_FILENAME=os.path.join("record",OUTPUT_FILENAME)
	trainedfilelist = open("record.txt", 'a')
	trainedfilelist.write(OUTPUT_FILENAME+"\n")
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(Recordframes))
	waveFile.close()
    


# import model
flasklink = Flask(__name__)
# model = pickle.load(open('MachineLearning.pkl','rb'))


@flasklink.route('/')
def index():
  return render_template('Home.html',custom_css = "home")

@flasklink.route('/link',  methods=['GET', 'POST'])
def link():
    record_audio_test()
    return render_template('Home.html', custom_css = "home")


# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=9000)

