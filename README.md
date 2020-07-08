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
    
    a.Using Spacy POS Tagger, parts of speech was extracted.

C. Unsupervised Classification:
    Establish rules for classifying sentences into actionable and non-actionable.
