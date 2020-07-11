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
    
 d) Classification Supervised:
 
    Useing BERT Classifier to classify the sentences into ACTIONABLE or NON-ACTIONABLE with training data from unspervised Classification

STEPS TO EXECUTE : 

a. Install the required set of python packages: spacy,numpy,pandas,tensorflow,tqdm 

b. Config file in json which points to input dataset,train,test files and url of basepath where the output will be stored.

c. Create an object of Class Assignment with url of the config file.

d. Download SPACY MODEL for Parts Of Speech Tagging , Dependency parser and Lemmatization.

e. run the Data Process method from the object.

f. Run the SentenceClassification (Unsupervised) from the object.

e For Supervised Classification: Go To https://colab.research.google.com/drive/11ZOIiF4vDVev9n47v4nJ4CkakL19E65A#scrollTo=HAah8tMgosYz
   
   * We obtain Train and Test from the results of  Unsupervised learning using train_test_split strategy.
   
   * Upload the TRAIN and TEST files using the cell for uploading in google colab link                 
   
   *  Run ALL the cells in Google colab link https://colab.research.google.com/drive/11ZOIiF4vDVev9n47v4nJ4CkakL19E65A#scrollTo=HAah8tMgosYz :
   
   * IPython script will download BERT MODEL from "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
   
   * The steps involved in Supervised Classification of sentences are as follows:
       
       +  install the packages bert-for-tf2 and sentencepiece
       
       +  invoking Bert Embedding layer from https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1 with Trainable to True for tuning BERT.
       
       +  Inserting [CLS] and Appending [SEP] to each sentence in Data Set.
       
       +  Using Bert Tokenizer, tokenizing the words to word_ids  
       
       +  Adding Masks and Segments for each sentence.
       
       +  Defining a model with Bert Layer Embedding + Global Average Pooling + Dropout + Dense Layer with an output of sigmoid for binary classification
       
       +  Using Adam Optimizer for optimising the model.
       
       +  Output label is indexed to 1 for ACTIONABLE and 0 for NON-ACTIONABLE 
       
       + Fitting the model to indexed word_ids,segments and masks obtained for each sentence from the Train dataframe along with 0-1 indexed Labels

CHALLENGES IN DATA PREPARATION:  

   a. Sentence and field separation. Fields such as From / To are separated by '\n' whereas sentences are separated by period. Along with that period also comes in salutations ('Mr.' and  sequences indices such as '1.' )
   
   b. mail ids in text
   
   c. Forwarding text of a mail
   
   d. HTML links and html tags in mail
   
   e.  different types of marking of EOM. " ******* " / " _______" /-=-=-=-=-=-=-=-=-=-=- "
   
   f. remnants and residuals of solutions to above issues.
   
   g. Verifying the sentences above 100k sentences
   

REFERENCES :

THANKS TO TEXT CLASSIFICATION KERNELS FROM KAGGLE WHICH USES BERT .  
