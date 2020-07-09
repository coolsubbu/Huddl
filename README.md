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
    
     
C. Unsupervised Classification:
    Establish rules for classifying sentences into actionable and non-actionable.

Challenges in Data Preparation:
   a. Sentence and field separation. Fields such as From / To are separated by '\n' whereas sentences are separated by period. Along with that period also comes in salutations ('Mr.' and  sequences indices such as '1.' )
   b. mail ids in text 
   c. Forwarding text of a mail
   d. HTML links and html tags in mail
   e.  different types of marking of EOM. " ******* " / " _______" /-=-=-=-=-=-=-=-=-=-=- "
   f. remnants and residuals of solutions to above issues.
   
