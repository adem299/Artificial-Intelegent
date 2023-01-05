from flask import Flask, render_template,request, url_for
import re
from nltk import WordNetLemmatizer
import pickle

wo = WordNetLemmatizer()

app = Flask(__name__)

def preprocess(data):
    #preprocess
    a = re.sub('[^a-zA-Z]',' ',data)
    a = a.lower()
    a = a.split()
    a = [wo.lemmatize(word) for word in a ]
    a = ' '.join(a)
    return a


tfidf_vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model =  pickle.load(open('prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/#about')
def about():
    return render_template('home.html')

@app.route('/test-by-sentence')
def sentence():
    return render_template('bysentence.html')

@app.route('/test-by-image')
def image():
    return render_template('byimage.html')



@app.route('/predict', methods= ['POST'])
def predict():
    msg = request.form['mood_pred']
    a = preprocess(msg)

    result = model.predict(tfidf_vectorizer.transform([a]))[0]
    return render_template('bysentence.html',pred = result)



if __name__ == '__main__':
    app.run(debug=True)