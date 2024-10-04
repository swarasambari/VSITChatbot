import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the Flask app
app = Flask(__name__)

# Load model and other necessary files
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = pickle.load(open('intents.json', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Define function to preprocess the user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert user input into bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict the intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get a response from the chatbot
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return i['responses']

# Define Flask route to handle chat messages
@app.route("/")
def index():
    return render_template("index.html")  # This will point to an HTML file for your frontend

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    ints = predict_class(message, model)
    response = get_response(ints, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
