# Importing libraries
from flask import Flask, render_template, request
import pickle

# Loading the mnb model and the tfidf model
model = pickle.load(open('spam_mnb.pkl','rb'))
tfidf = pickle.load(open('tfidf_transformed.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = tfidf.transform(data).toarray()
        prediction = model.predict(vector)
        return render_template('result.html', prediction=prediction)
    
if __name__=='__main__':
    app.run(debug = True)