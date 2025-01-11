'''import streamlit as st
import pickle

model=pickle.load('model.pkl', 'rb')
vectorizer=pickle.load('vectorizer.pkl', 'rb')

st.title("Email spam classification !!")
st.write("This will tell you the email sent to you is spam or not spam")

user_input=st.text_area("Enter the email to be tested" , height=150)

if st.button("Classify"): 
    if user_input :
        data= [user_input]
        vectorized_data = cv.transform(data).toarray()
        result=model.predict(vectorized_data)
        if result[0]==0:
            st.write("This email is not spam . \n Don't worry !! ^ ^")
        else:
            st.write("This email is a spam . \n BEWARE !! ^ ^") 
    else:
        st.write("Please type the email to classify")  '''      
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")

    

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
