from flask import Flask, request, jsonify
import json
import nltk
from tensorflow.keras.models import load_model
import numpy as np
import random
import pickle

# Load trained model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

app = Flask(__name__)

# Preprocessing and prediction functions (same as your chatgui.py)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Route to handle chatbot queries
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    ints = predict_class(user_message)
    response = get_response(ints, intents)
    return jsonify({"response": response})

# Home route to verify the API is working
@app.route("/", methods=["GET"])
def home():
    return "VSIT Chatbot is up and running!"

if __name__ == "__main__":
    app.run()
