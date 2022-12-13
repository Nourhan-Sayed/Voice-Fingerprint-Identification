#import  libraries---------------------------------------------------------
from flask import Flask , render_template
import numpy as np
from functions import *

flasklink = Flask(__name__)

@flasklink.route('/')
def index():
	return render_template('Home.html',custom_css = "home")

@flasklink.route('/link',  methods=['GET', 'POST'])
def link():
    sentence_flag = False
    record_audio_test()
    team_flag ,member_name = test_model("people","trained_models\Team_Verification")
    print(member_name)
    if (team_flag) :
        # _, sentence = test_model("sentence")
        _, sentence = test_model("sentence",f"trained_models\Sentence_Verification\{member_name}")
        print(sentence)
        print("Access Allowed")
        if sentence == f"{member_name}_Open_The_Door" :
            print("OPEN")
            sentence_flag =True
            print(sentence_flag)
        else :
            print("Wrong Word")
            sentence_flag =False
    else :
        print("Access Denied")
    return render_template('Home.html',Team_Flag=team_flag, Member_Name=member_name,Sentence_Flag=sentence_flag, custom_css = "home")

# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=50000)