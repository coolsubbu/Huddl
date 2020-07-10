# Huddl
A Project for classification of sentences as actionable or non-actionable

This project consists of both supervised and unsupervised methods of classifying a sentence into actionable or non-actionable.

The Pipeline consists of 

a)Data Preparation 

b)Feature Extraction 

c)Classification

A. Data Preparation:

   a.Read the csv
   
   b.For each mail, remove the fields of the email such as From,To
   
   c.Extract the mail content 
   
B.Feature Extraction:
    
    Using spacy tagger to find out Parts of speech and dependency parser in addition to lemmatization.

STEPS TO EXECUTE : 
a. Config file in json which points to input dataset and url of basepath where the output will be stored.
b. Create an object of Class Assignment with url of the config file.


         
C. Unsupervised Classification:
    Establish rules for classifying sentences into actionable and non-actionable.

Challenges in Data Preparation:
   
   a. Sentence and field separation. Fields such as From / To are separated by '\n' whereas sentences are separated by period. Along with that period also comes in salutations ('Mr.' and  sequences indices such as '1.' )
   
   b. mail ids in text
   
   c. Forwarding text of a mail
   
   d. HTML links and html tags in mail
   
   e.  different types of marking of EOM. " ******* " / " _______" /-=-=-=-=-=-=-=-=-=-=- "
   
   f. remnants and residuals of solutions to above issues.
   
   g. Verifying the sentences above 100k sentences
   

STEPS TO EXECUTE : 

a. Config file in json which points to input dataset and url of basepath where the output will be stored.

b. Create an object of Class Assignment with url of the config file as the parameter.

c.  call function data_preprocess to obtain to preprocess input data 

d.  call the function Sentence_Classification to classify the preprocessed sentences into ACTIONABLE or non ACTIONABLE.          
