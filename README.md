# Huddl project : classification of sentences as actionable or non-actionable

This project consists of both supervised and unsupervised methods of classifying a sentence into actionable or non-actionable.

Pipeline:

a)data preparation 

b)feature extraction 

c)classification unsupervised

d) classification supervised

e) api creation

f) api verification

A. data preparation:

   a. read csv
   
   b. for each mail, remove the fields of the email such as from, to
   
   c.extract the mail content 
   
B.feature extraction:
    
    using spacy tagger to find out parts of speech and dependency parser in addition to lemmatization.

c) classification unsupervised:
    
    using manual defined rules to classify the sentences into actionable and unactionable
 
 d) classification supervised ( long short term memory network  and convolutional neural network ):
     
     using lstm + cnn classifier to classify the sentences into actionable or non actionable with training data from unsupervisedd classification.
     
steps to execute : 

a. install the required set of python packages: pip install -r requirements.txt  

b. config file in json which points to input dataset,train,test files and url of basepath where the output will be stored.

c. create an object of class assignment with url of the config file.

d. download spacy model for parts of speech tagging , dependency parser and lemmatization.

e. run the data process method from the object.

f. run the sentence classification ( unsupervised ) from the object.

g. run the supervised classification from the object.

    lstm + cnn supervised sentence classification :
    
    * result of the unsupervised classification and actions file are both imbibed into training file.
    
    * the training file is string indexed using a tokenizer and split into train / test using train_test_split strategy.
    
    * a sequential keras model with embedding + convolution + lstm (100 units) + dense + sigmoid is constructed 
    
    * model is fit to train data and evaluated with test data.

h. run the api.py to start the server.

i) go to http:/localhost:5000/api/sentence_classify?sentence=<<sentence>>
   
loading the model to predict:

It is a keras model . It can be loaded by " keras.models.load_model(<<my_model>>) "


challenges in data preparation:  

   a. sentence and field separation. fields such as from / to are separated by '\n' whereas sentences are separated by period. 
       along with that period also comes in salutations ('Mr.' and  sequences indices such as '1.' )
   
   b. mail ids in text
   
   c. forwarding text of a mail
   
   d. html links and html tags in mail
   
   e. different types of marking of eom . " ******* " / " _______" /-=-=-=-=-=-=-=-=-=-=- "
   
   f. remnants and residuals of solutions to above issues.
   
   g. verifying the sentences above 100k sentences
 

results on test set from lstm + cnn sentence classifier:

total number of records: ~145k  

actionable:~60k         

non actionable:~85k

precision = 0.96

recall = 0.97  

f1 = 2PR/(P+R) = 0.96

trained model: 

It is of size 56MB and highest github could handle is only 25MB. 

It is present in this google drive link : https://drive.google.com/file/d/1dv2pM2hgPI5kEwJFKMklAgF0NPnPKbhc/view?usp=drive_web

note: the model needs to be used along with the tokenizer to classify a sentence into actionable/non actionable.     


API: host model


![image](https://user-images.githubusercontent.com/1144567/87305515-41bb8100-c534-11ea-8e2f-931a974d544c.png)

![image](https://user-images.githubusercontent.com/1144567/87305806-b2fb3400-c534-11ea-972c-8a0bcb550d52.png)

steps to execute for api : 

a.go to api.py

b.load the model 

c.load the tokenizer

d. execute api.run() to run the server.

e.  go to browser and type http://localhost:5000/api/sentence_classify?sentence='please mail back'           


references :
         
1.https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
      
