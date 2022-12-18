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
    filename=record_audio_test()
    encoded_img_data = create_MFCC_img()
    team_flag ,member_name = test_model("people","trained_models\Team_Verification",filename)
    if member_name in ["Other_1","Other_2","Other_3"]:
        team_flag = False
    print(member_name)
    if (team_flag) :
        # _, sentence = test_model("sentence")
        _, sentence = test_model("sentence",f"trained_models\Sentence_Verification\{member_name}",filename)
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
    return render_template('Home.html',Team_Flag=team_flag, Member_Name=member_name,Sentence_Flag=sentence_flag,img_data= encoded_img_data.decode('utf-8'), custom_css = "home")

# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=9000)