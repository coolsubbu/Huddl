import keras
import pickle
from keras.preprocessing import sequence

toknisr=pickle.load(open('C:/Users/lenovo/Downloads/Huddl/tokenizer.h5','rb'))
modl1=keras.models.load_model('C:/Users/lenovo/Downloads/Huddl/model')
sentence="please mail back"
sent=toknisr.texts_to_sequences([sentence])
sent1=sequence.pad_sequences(sent,maxlen=50)
pred=modl1.predict_classes(sent1)
print(sent)
print(sent1)
print(pred[0][0])
