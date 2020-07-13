# Huddl
A Project for classification of sentences as actionable or non-actionable

This project consists of both supervised and unsupervised methods of classifying a sentence into actionable or non-actionable.

PIPELINE:

a)Data Preparation 

b)Feature Extraction 

c)Classification Unsupervised

d) Classification Supervised

A. Data Preparation:

   a.Read the csv
   
   b.For each mail, remove the fields of the email such as From,To
   
   c.Extract the mail content 
   
B.Feature Extraction:
    
    Using spacy tagger to find out Parts of speech and dependency parser in addition to lemmatization.

c) Classification Unsupervised:
    
    Using manual defined rules to classify the Sentences into ACTIONABLE and UNACTIONABLE
 
 d) Classification Supervised (LSTM and Convolutional NN):
     
     Using LSTM+CNN Classifier to classify the sentences into ACTIONABLE or NON-ACTIONABLE with training data from Unsupervisedd Classification.
     
STEPS TO EXECUTE : 

a. Install the required set of python packages: pip install -r requirements.txt  

b. Config file in json which points to input dataset,train,test files and url of basepath where the output will be stored.

c. Create an object of Class Assignment with url of the config file.

d. Download SPACY MODEL for Parts Of Speech Tagging , Dependency parser and Lemmatization.

e. run the Data Process method from the object.

f. Run the SentenceClassification (Unsupervised) from the object.

g. Run the Supervised Classification from the object.

    LSTM+CNN Supervised Sentence Classification :
    
    * Result of the Unsupervised Classification and Actions file are both imbibed into Training file.
    
    * The Training File is string indexed using a Tokenizer and spllit into Train / Test using train_test_split strategy.
    
    * a Sequential Keras model with Embedding + Convolution + LSTM (100 units) + Dense +Sigmoid is constructed 
    
    * model is fit to train data and evaluated with Test Data.
    
LOADING THE MODEL TO PREDICT:

It is a Keras model . It can be loaded by " keras.models.load_model(<<my_model>>) "


CHALLENGES IN DATA PREPARATION:  

   a. Sentence and field separation. Fields such as From / To are separated by '\n' whereas sentences are separated by period. Along with that period also comes in salutations ('Mr.' and  sequences indices such as '1.' )
   
   b. mail ids in text
   
   c. Forwarding text of a mail
   
   d. HTML links and html tags in mail
   
   e.  different types of marking of EOM. " ******* " / " _______" /-=-=-=-=-=-=-=-=-=-=- "
   
   f. remnants and residuals of solutions to above issues.
   
   g. Verifying the sentences above 100k sentences
 

RESULTS on TEST SET FROM LSTM + CNN Sentence Classifier:

TOTAL NUMBER OF RECORDS: ~145k  

ACTIONABLE:~60k         

NON-ACTIONABLE:~85k

PRECISION = 0.96

RECALL = 0.97  

F1 = 2PR/(P+R) = 0.96

TRAINED MODEL: It is of size 56MB and highest github could handle is only 25MB. It is present in this Google drive link :
https://drive.google.com/file/d/1dv2pM2hgPI5kEwJFKMklAgF0NPnPKbhc/view?usp=drive_web

NOTE: The model needs to be used along with the tokenizer to classify a sentence into Actionable/NON-Actionable.     


API: Converted the model into an API

![image](https://user-images.githubusercontent.com/1144567/87305515-41bb8100-c534-11ea-8e2f-931a974d544c.png)

![image](https://user-images.githubusercontent.com/1144567/87305806-b2fb3400-c534-11ea-972c-8a0bcb550d52.png)

STEPS TO EXECUTE FOR API:

a.Go to api.py

b. Load the Model 

c. Load the Tokenizer

d. execute api.run() to run the server.

e.  go to browser and type http://localhost:5000/api/sentence_classify?sentence='please mail back'           

REFERENCES :
         
1.https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
