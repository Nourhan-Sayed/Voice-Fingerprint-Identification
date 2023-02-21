#import  libraries---------------------------------------------------------
from flask import Flask , render_template
import numpy as np
from functions import *

flasklink = Flask(__name__)

# -------------------------------------APIs----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------------------------------------------------#
# API description:
#       Fuction: Render the home page
#       Return: Home page
@flasklink.route('/')
def index():
	return render_template('Home.html',custom_css = "home")

# ----------------------------------------------------------------------------------------------------------------------#
# API description:
#       Fuction: take the audio file from user and use the gmm models
#       to predict the speaker and the word and decide to give access or not.
#       Return: Home page with the result of the prediction and dynamic graphs
@flasklink.route('/link',  methods=['GET', 'POST'])
def link():
    sentence_flag = False
    # record audio from user and save it in a file
    filename=record_audio_test()
    
    # test Team_Verification model to predict the speaker and its access
    team_flag ,member_name ,log_likelihood , speakers = test_model("people","trained_models\Team_Verification",filename)
    if member_name in ["Other_1","Other_2","Other_3"]:
        team_flag = False
    # print(member_name)
    # if the speaker is in the team then test the word
    if (team_flag) :
        # test Sentence_Verification model to predict the word and its access
        _, sentence,_,_ = test_model ("sentence",f"trained_models\Sentence_Verification\{member_name}",filename)
        # print(sentence)
        print("Access Allowed")
        if sentence == f"{member_name}_Open_The_Door" :
            sentence_flag =True
            # print("OPEN")
            # print(sentence_flag)
        else :
            sentence_flag =False
            # print("Wrong Word")
    else :
        print("Access Denied")

    # Dynamic graphs
    # create the mfcc spectogram image
    encoded_img_data = create_MFCC_img(filename)
    # create the normal distribution image
    create_normal_img(filename)
    # create the scores image
    create_scores_img(log_likelihood , speakers)

    # render the home page with the result of the prediction and dynamic graphs
    return render_template('Home.html',Team_Flag=team_flag, Member_Name=member_name,Sentence_Flag=sentence_flag,img_data= encoded_img_data.decode('utf-8'), custom_css = "home")

# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=9000)