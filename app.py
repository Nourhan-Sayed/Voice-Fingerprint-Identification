#import  libraries
from flask import Flask , render_template
import pickle
import numpy as np


# import model
flasklink = Flask(__name__)
# model = pickle.load(open('MachineLearning.pkl','rb'))

# Home page
@flasklink.route("/") 
def Home():

    return render_template("Home.html", pagetitle = "home", custom_css = "home")  


# run our programe
if __name__== "__main__":  #for make this content appear when file open directly not importent from another file
    flasklink.run(debug=True, port=9000)

