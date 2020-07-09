import pandas as pd
import re
import spacy
import json
import time

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
                 
        print(time.asctime())       
        for i,v in self.input_df.iterrows():

          v_dict={}

          mail_strings=v['message'].split('\n')

          #removing fields and extracting only  mail text content
     
          mail_strings=[s for s in mail_strings if not re.search(r'\w\:',s)]

          if i%250==0:
               print('0:'+str(mail_strings)+'\n')

          mail_strings=[s for s in mail_strings if '----- Forwarded by' not in s]

          mail_strings=[re.sub(r'\<(.*?)\>',' ',s) for s in mail_strings] 
                               
          mail_strings=[ s for s in mail_strings if not re.search(r'(\@ECT|\@enron.com)',s)]

          mail_strings=[re.sub('\"\"+','\"',s) for s in mail_strings]  
                      
          mail_strings=[ re.sub(r'(\<|\>|\-\-\-+|\\t)','',s)   for s in mail_strings]

          mail_strings=[ re.sub(r'\[image\]','',s) for s in mail_strings]

          mail_strings=[ re.sub(r'(\[\]\[\])+','',s) for s in mail_strings]
          
          if i%250==0:
               print('mail_strings:'+str(mail_strings)+'\n')

          mail=' '.join(mail_strings)

       
          # splitting mail into sentences
          mail=re.sub(r'(Mr|Ms|Dr|Prof|\d)\.','.',mail)

          mail=re.sub(r'(\.\.+)','.',mail)
          
          sentences=re.split(r'((?=\D)\.\s\s*|\?\s+|\!\s+)',mail) 

          sentences=[s for s in sentences if not re.match(r'(\.|\?|\!)\s+',s)]

          sentences=[re.sub('\s+',' ',sentence) for sentence in sentences]
          
          sentences=[s.lower() for s in sentences ]

          sentences=[re.sub(r'\=(\s|\d+)','',s) for s in sentences]

          sentences=[re.sub(r'\=(\d\w)','',s) for s in sentences]
          
          sentences=[re.sub('\s(.*?)\@(.*?)\s',' ,',s) for s in sentences]
          
          sentences=[s for s in sentences if not re.match(r'^\s*$',s)]

          sentences=[re.sub('(\+\++|\=\=+|--+|\~\~+|\?\?+|\*\*+|__+|\/\/+|\#\#+)',' ',s) for s in sentences]

          sentences=[re.sub('(\|\s)+',' ',s) for s in sentences]

          sentences=[re.sub('(-\=-)+',' ',s) for s in sentences]

          sentences=[re.sub('(\=\s+\=)+',' ',s) for s in sentences]
        
          sentences=[re.sub('(\#\s)+',' ',s) for s in sentences]

          sentences=[re.sub(r'\[image\]','',s) for s in sentences]

          sentences=[re.sub(r'\(e-mail\)','',s) for s in sentences]

          sentences=[re.sub(r'\&nbsp','',s) for s in sentences]

          sentences=[re.sub(r'dd+','',s) for s in sentences]

          sentences=[re.sub(r'\"\"','\"',s) for s in sentences]
          
          sentences=[re.sub(r', (, )+',',',s) for s in sentences]

          sentences=[re.sub(r'\!\!+','!',s) for s in sentences]
          
          sentences=[re.sub(r'\s+',' ',s) for s in sentences]

          sentences=[re.sub(r'\"\"','\"',s) for s in sentences]
          
          sentences=[s for s in sentences if len(s.split(' '))>2]

          if i%250==0:
             print(str(sentences)+'\n')
          
          for sentence in sentences:
               sentence_df_dict={}
               if len(sentences)>2:
                   sentence_df_dict['sentence']=sentence
                   sentence_df_dict['length']=len(sentence.split(' '))
                   #sentence_df_dict['raw']=mail_strings
                   sentence_df_list.append(sentence_df_dict)
          
          if i1>20000:
            break
          i1=i1+1

        self.sentence_df=pd.DataFrame(sentence_df_list,columns=list(sentence_df_list[0].keys()))

        self.sentence_df.to_csv(self.basPath+"sentences_571.csv")

          
    def SentenceClassification(self):

        sentence_cls_df_dict={}
        sentence_cls_df_list=[]

        #self.sentence_df=pd.read_csv(self.basPath+'sentences_s.csv')
        
        for i,v in self.sentence_df.iterrows():
            sentence=v['sentence']
            v['class']='empty'
            if 'your thoughts' in sentence or 'kindly' in sentence or 'as discussed' in sentence or 'needs to' in sentence:
                v['class']='ACTIONABLE'
                
            if 'better if' in sentence or 'need' in sentence or 'could use' in sentence:
               v['class']='ACTIONABLE' 
               #print('ACTIONABLE')
              
            spacy_pos=self.nlp(sentence)
            #print(sentence)
            DEP=[tok.dep_ for tok in spacy_pos]
            TEXT=[tok.text for tok in spacy_pos]
            
            LEMMA=[tok.lemma_ for tok in spacy_pos]
                
            POS=[tok.pos_ for tok in spacy_pos]
            TAG=[tok.tag_ for tok in spacy_pos]
            v['TAG']=TAG
            v['POS']=POS
            v['LEMMA']=LEMMA
            v['DEP']=DEP
            #print('text'+str(TEXT))
            #print('pos'+str(POS))
            #print('tag'+str(TAG))
           
            #for token in spacy_pos_:
            #    if token.pos_ not in Pos_dict:
            #        Pos_dict[token.pos_]=[]
            #       Pos_dict[token.pos_].append(j1)
               
            k1=0                                        
            for token in spacy_pos:
               #print(token.text, token.pos_, token.tag_, token.dep_)
               if 'suggest' in LEMMA or 'please' in LEMMA or 'kindly' in TEXT or 'propose' in LEMMA or 'need' in LEMMA:
                    v['class']='ACTIONABLE' 
                    #print('ACTIONABLE')
               if spacy_pos[0].tag_=='VB':
                    if spacy_pos[1].pos_=='DET' :
                         #print('ACTIONABLE')
                         v['class']='ACTIONABLE'
                    elif spacy_pos[1].pos_=='NOUN':
                         #print('ACTIONABLE')
                         v['class']='ACTIONABLE'

               #send the ppt or send ppt 
               if spacy_pos[k1].tag_=='VB':
                    
                  if k1+1<len(spacy_pos) and spacy_pos[k1+1].pos_=='NOUN':
                       #print('ACTIONABLE')
                       v['class']='ACTIONABLE'
                  if k1+1<len(spacy_pos) and spacy_pos[k1+1].pos_=='DET':
                       if k1+2<len(spacy_pos) and spacy_pos[k1+2].pos_=='NOUN':
                         #print('ACTIONABLE')
                         v['class']='ACTIONABLE'
                  if k1+1<len(spacy_pos) and (spacy_pos[k1+1].pos_=='PRON' or spacy_pos[k1+1].pos_=='IN' or spacy_pos[k1+1].pos_=='TO'):
                         v['class']='ACTIONABLE'
                         

               #can you send 
               if spacy_pos[k1].tag_=='MD':
                    
                    if k1+1<len(spacy_pos) and spacy_pos[k1+1].tag_=='PRP':
                         if k1+2<len(spacy_pos) and spacy_pos[k1+2].pos_=='VERB':
                              #print('ACTIONABLE')
                              v['class']='ACTIONABLE'
               k1=k1+1
            if v['class']=='empty':
              v['class']='NON-ACTIONABLE'
            sentence_cls_df_list.append(v)
        sentence_cls_df=pd.DataFrame(sentence_cls_df_list,columns=list(sentence_cls_df_list[0].keys()))
        sentence_cls_df.to_csv(self.basPath+'sentence_classified_sb.csv')
        sentence_cls_file_df=pd.DataFrame(sentence_cls_df_list,columns=['sentence','class'])
        sentence_cls_file_df.to_csv(self.basPath+'sentence_classified_file_1.csv')
                                    


             
if __name__=='__main__':
     Assign1=Assignment("C:/Users/lenovo/Downloads/Assignment_1/config.json")
     Assign1.data_preprocess()
     Assign1.SentenceClassification()
