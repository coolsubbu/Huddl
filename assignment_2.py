import pandas as pd
import re
import spacy
import json

class Assignment:

    def __init__(self,config_file_url):

        print('loading models and input csv')

        print("loading spacy nlp  and tagger model ")

        self.nlp= spacy.load("en_core_web_sm")
        self.tagr=spacy.load('en')
        print("loaded spacy nlp tagger model ")

        print('reading input csv and sanity check')

        self.config_file_handle=open(config_file_url,"r",encoding='utf-8')
         
        print(config_file_url)

        assert self.config_file_handle, ' Could not open config File'
    
        self.config= json.load(self.config_file_handle)

        print('loading input dataset')

        self.input_data=pd.read_csv(self.config["DataURL"],keep_default_na=False,encoding='latin-1')

        self.input_df =self.input_data

        assert not self.input_data.empty, "Could not read input dataset or input_data empty" 
        
        print('loaded input csv')

        self.sentence_df={}   
                                               
        self.basPath=self.config["BASEPATH"]
        
    def data_preprocess(self):
       
        i1=0
        sentence_df_list=[]
        sentence_df_dict={}
        
                
        for i,v in self.input_df.iterrows():

          v_dict={}

          mail_strings=v['message'].split('\n')

          #removing fields and extracting only  mail text content
     
          mail_strings=[s for s in mail_strings if not re.search(r'\w\:',s)]

          print('0:'+str(mail_strings)+'\n')

          mail_strings=[s for s in mail_strings if '----- Forwarded by' not in s]
     
          mail_strings=[ s for s in mail_strings if not re.search(r'(\@ECT|\@enron.com)',s)]
     
          mail_strings=[ re.sub('(\>|\-\-\-+|\\t)','',s)   for s in mail_strings]
     
          print('mail_strings:'+str(mail_strings)+'\n')

          mail=' '.join(mail_strings)

          # splitting mail into sentences
          mail=re.sub(r'(Mr|Ms|Dr|Prof|\d)\.','.',mail)
          
          sentences=re.split(r'((?=\D)\.\s\s*|\?\s+|\!\s+)',mail) 

          sentences=[s for s in sentences if not re.match(r'(\.|\?|\!)\s+',s)]

          sentences=[re.sub('\s+',' ',sentence) for sentence in sentences]

          sentences=[s.lower() for s in sentences ]

          sentences=[s for s in sentences if not re.match(r'^\s*$',s)]
          sentence=[s for s in sentences if len(s)>2]
          
          print(str(sentences)+'\n')
          
          for sentence in sentences:
               sentence_df_dict={}
               sentence_df_dict['sentence']=sentence
               sentence_df_list.append(sentence_df_dict)
          
          if i1>200:
            break
          i1=i1+1

        self.sentence_df=pd.DataFrame(sentence_df_list,columns=list(sentence_df_list[0].keys()))

        self.sentence_df.to_csv(self.basPath+"sentences.csv")

    '''      
    def SentenceClassification(self):

         
        for i,v in sentence_df.iterrows():
           sentence=v['sentence']
           
           if 'your thoughts' in sentence or 'kindly' in sentence:
              print('ACTIONABLE')
           spacy_pos=nlp(sentence)
           print(sentence)
           TEXT=[tok.text for tok in spacy_pos]
           LEMMA=[tok.lemma_ for tok in spacy_pos]
           POS=[tok.pos_ for tok in spacy_pos]
           TAG=[tok.tag_ for tok in spacy_pos]
           print('text'+str(TEXT))
           print('pos'+str(POS))
           print('tag'+str(TAG))
           
           #for token in spacy_pos_:
           #    if token.pos_ not in Pos_dict:
           #        Pos_dict[token.pos_]=[]
           #       Pos_dict[token.pos_].append(j1)
               
           k1=0                                        
           for token in spacy_pos:
               print(token.text, token.pos_, token.tag_, token.dep_)
               if 'suggest' in LEMMA or 'please' in LEMMA or 'kindly' in TEXT or 'propose' in LEMMA:
                    print('ACTIONABLE')
               if spacy_pos[0].tag_=='VB':
                    if spacy_pos[1].pos_=='DET' :
                         print('ACTIONABLE')
                    elif spacy_pos[1].pos_=='NOUN':
                         print('ACTIONABLE')

               #send the ppt or send ppt 
               if spacy_pos[k1].tag_=='VB':
                    
                  if k1+1<len(spacy_pos) and spacy_pos[k1+1].pos_=='NOUN':
                       print('ACTIONABLE')
                  if k1+1<len(spacy_pos) and spacy_pos[k1+1].pos_=='DET':
                       if k1+2<len(spacy_pos) and spacy_pos[k1+2].pos_=='NOUN':
                         print('ACTIONABLE')

               #can you send 
               if spacy_pos[0].tag_=='MD':  
                    if spacy_pos[1].pos_=='PRON':
                         if spacy_pos[2].pos_=='VERB':
                              print('ACTIONABLE')
               k1=k1+1
    ''' 
if __name__=='__main__':
     Assign1=Assignment("C:/Users/lenovo/Downloads/Assignment_1/config.json")
     Assign1.data_preprocess()
