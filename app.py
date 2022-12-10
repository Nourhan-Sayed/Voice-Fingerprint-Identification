#import  libraries---------------------------------------------------------
from flask import Flask , render_template
import pickle
import numpy as np
import os
import wave
import pyaudio

from functions import *


# import model
flasklink = Flask(__name__)
# model = pickle.load(open('MachineLearning.pkl','rb'))
sentence_flag = False
# team_flag= False

@flasklink.route('/')
def index():
	return render_template('Home.html',custom_css = "home")

@flasklink.route('/link',  methods=['GET', 'POST'])
def link():

    record_audio_test()

    team_flag ,member_name = test_model("people")
    if (team_flag) :
        print("Access Allowed")
        _, sentence = test_model("sentence")
        if sentence == "Open_The_Door_Omar" :
            print("OPEN")
            sentence_flag =True
        elif sentence == "Close_The_Door_Omar" :
            print("Close")
            sentence_flag =False
        else :
            print("Nothing")
            sentence_flag =False
    else :
        print("Access Denied")
    return render_template('Home.html', custom_css = "home")


# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=9000)

