import flask
from flask import request, jsonify
from assignment_2 import Assignment as assign
import pickle
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import time
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics as sm
import flask
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Sentence Classification</h1>
<p>A prototype API for classifying sentence into actionable or non-actionable.</p>'''


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/api/sentence_classify',methods=['GET'])
def classify():                                                  

 
    #Assign=assign(r'C:/ Users/ lenovo/ Downloads/ Assignment_1/ config.json')

    #model1=keras.models.load_model('C:/ Users/ lenovo/ Downloads/ Huddl/ model')
                                                
    #tokenizer=pickle.load(open(r'C:/ Users/ lenovo/ Downloads/ Huddl/ tokenizer.h5','rb'))

    
    if 'sentence' in request.args:
        sentence = request.args['sentence']
    else:
        return "Error: No sentence provided. Please specify a sentence."
    
    toknisr=pickle.load(open(r'C:/Users/lenovo/Downloads/Huddl/tokenizer.h5','rb'))
    #modl1=keras.models.load_model(r'C:/Users/lenovo/Downloads/Huddl/model')
    #sentence="please mail back"
    sent=toknisr.texts_to_sequences([sentence])
    sent1=sequence.pad_sequences(sent,maxlen=50)
    pred=model1.predict_classes(sent1)
    print(sent)
    print(sent1)
    print(pred)
    string= "<H1> "+sentence+"<H1>" +"<P>"+str(sent) +"<P>"+"<H1>"+str(sent1)+"<H1>"
    
    if pred[0][0]==1:
        return string+'<H1> ACTIONABLE<H1>'
    if pred[0][0]==0:
        return string+'<H1> NON-ACTIONABLE <H1>'
    
    
    #return "<H1> "+sentence+"<H1>" +"<P>"+str(sent) +"<P>"+"<H1>"+str(sent1)+"<H1>"


if __name__=="__main__":

    #Assign=assign(r'C:/Users/lenovo/Downloads/Assignment_1/config.json')
                    
    #tokenizer=pickle.load(open(Assign.basPath+'tokenizer.h5',"rb"))

    model1=keras.models.load_model('C:/Users/lenovo/Downloads/Huddl/model')

    app.run()
