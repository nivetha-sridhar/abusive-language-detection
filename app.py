from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words("english"))
app = Flask(__name__)
def preprocess_texts(text):

    #converting all tweest to lowercase
    text =  text.lower()

    #removing urls using regx
    text = re.sub(r"http\S+|www\S+|https\S+","",text , flags = re.MULTILINE) #MULTILINE flag checks for the regx at the start of the string or even inside the string
      
    #remove punctuations
    text =  text.translate(str.maketrans("","",string.punctuation))

    #remove # and @
    text = re.sub(r"\@\w+|\#" , "",text)

    #remove stopwords
    text_tokens = word_tokenize(text)
    filtered_words = [word for word in text_tokens if word not in stop_words]

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(words , pos = "a") for words in filtered_words]

    return " ".join(lemma_words) 

df = pd.read_excel("D:\youtube_Comments.xlsx")
df =  df.iloc[:,2:5]
df = df.drop("IsToxic",axis=1)
Z = df["Text"].astype("string")
df["text_preprocessed"] = Z.apply(preprocess_texts)

print(df["text_preprocessed"].head(10))
df.head(10)

x = df["text_preprocessed"]
y = df["IsAbusive"]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,random_state = 42,test_size = 0.2)  

cv = TfidfVectorizer() 

X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

model = LogisticRegression(class_weight="balanced") #76
model.fit(X_train_vec,Y_train)

@app.route("/")
def msg():
    return render_template("msg.html", message="")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text to predict from the request
    text = request.form['text']
    text = preprocess_texts(text)
    # Vectorize the text using the trained vectorizer
    X_pred = cv.transform([text])
    
    # Make a prediction using the trained model
    prediction = model.predict(X_pred)
    
    # Return the prediction as a JSON object
    # return jsonify({'prediction': bool(prediction[0])})
    return str(bool(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)