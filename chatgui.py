#please read the Run_ChatBot_Guide before running this file

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import random
import tkinter
from tkinter import *

# Load model and data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load keywords from keywords.json
keywords = json.loads(open('keywords.json').read())

# Function to clean up input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Predict class of the input sentence
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

# Get response based on intent
def getResponse(ints, intents_json):
    if len(ints) == 0:
        return "I'm sorry, I didn't understand that."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = None
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    if result is None:
        return "I'm sorry, I didn't find a response for that."
    
    return result

# Check for keyword responses and return corresponding intent
def get_keyword_intent(msg):
    for keyword, intent in keywords.items():
        if keyword.lower() in msg.lower():
            print(f"Keyword '{keyword}' detected, mapping to intent '{intent}'")
            return intent
    return None

# Check for keyword responses first
def get_keyword_response(msg):
    for keyword, response in keywords.items():
        if keyword.lower() in msg.lower():
            return response
    return None

# Get response from intent if no keyword match
def get_intent_response(msg, model, intents_json):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents_json)
    return res

# Main chatbot response function
def chatbot_response(msg):
    keyword_response = get_keyword_response(msg)
    if keyword_response:
        return keyword_response
    return get_intent_response(msg, model, intents)

# Suggest questions based on user input
def suggest_questions(event):
    typed_text = EntryBox.get("1.0", 'end-1c').strip()
    suggestions = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if typed_text.lower() in pattern.lower():
                suggestions.append(pattern)

    suggestions = suggestions[:8]

    for widget in suggestion_frame.winfo_children():
        widget.destroy()

    if suggestions:
        title_label = Label(suggestion_frame, text="Suggested Questions", font=("Arial", 10, 'bold'), fg="#333333", bg="white")
        title_label.pack(pady=(0, 5))

        instruction_label = Label(suggestion_frame, text="Click on any suggested question to get an answer.",
                                  font=("Arial", 9), bg="white", fg="#666666", wraplength=240)
        instruction_label.pack(pady=(0, 5))

    for suggestion in suggestions:
        suggestion_button = Button(suggestion_frame, text=suggestion, width=40, wraplength=300,
                                   command=lambda s=suggestion: insert_suggestion_and_send(s), bg="white")
        suggestion_button.pack(pady=2)

# Insert the selected suggestion into the EntryBox and auto-send
def insert_suggestion_and_send(suggestion):
    EntryBox.delete("0.0", END)
    EntryBox.insert(END, suggestion)
    send()

# Add placeholder text to EntryBox
def add_placeholder(event=None):
   if EntryBox.get("1.0", 'end-1c').strip() == "":
        EntryBox.insert("0.0", "Type keywords to get suggested questions related to your query...")
        EntryBox.config(fg="#999999", font=("Arial", 12))  # Set font to Arial, size 12

# Remove placeholder text when user starts typing
def remove_placeholder(event=None):
    if EntryBox.get("1.0", 'end-1c').strip() == "Type keywords to get suggested questions related to your query...":
        EntryBox.delete("0.0", END)
        EntryBox.config(fg="black")

# Modify the send function to handle the placeholder re-adding logic
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    
    # Check if the message is empty or just the placeholder
    if msg == "" or msg == "Type keywords to get suggested questions related to your query...":
        # Do not process the message if it's empty or only a placeholder
        return

    EntryBox.delete("0.0", END)

    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12))

    res = chatbot_response(msg)
    ChatLog.insert(END, "VSIT_BOT: " + res + '\n\n')

    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

# Initialize the Tkinter GUI
base = Tk()
base.title("VSIT ChatBot")
base.geometry("900x600")
base.resizable(width=FALSE, height=FALSE)

# Add header
header = Label(base, text="Vidyalankar School of Information Technology ChatBot", font=("Helvetica", 16, 'bold'), bg="white", fg="black")
header.pack(fill=X)

# Add description label
description = Label(
    base,
    text="Welcome to the VSIT ChatBot!\n"
         "Feel free to ask any questions you may have regarding Vidyalankar School of Information Technology. "
         "Whether you’re looking for information about courses, admissions, facilities, or placements, "
         "I’m here to help!",
    font=("Arial", 10),
    bg="white",
    fg="#333333",
    wraplength=700,
    justify="center"
)
description.pack(pady=(10, 5))

# Create Chat window with word wrapping
ChatLog = Text(base, bd=0, bg="white", height="15", width="255", font="Arial", wrap="word")
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message with adjusted height
EntryBox = Text(base, bd=0, bg="white", width="40", height="5", font="Arial", wrap="word")
EntryBox.bind("<KeyRelease>", suggest_questions)
EntryBox.bind("<FocusIn>", remove_placeholder)
EntryBox.bind("<FocusOut>", add_placeholder)

# Frame to hold suggestion buttons
suggestion_frame = Frame(base, bg="white")
suggestion_frame.place(x=590, y=130, width=270, height=390)

# Place components on the screen
scrollbar.place(x=555, y=130, height=400)
ChatLog.place(x=6, y=130, height=400, width=550)
EntryBox.place(x=128, y=520, height=30, width=480)  # Adjust height as needed
SendButton.place(x=6, y=520, height=30)

# Add watermark label
watermark = Label(base, text="Created by Swara Sambari", font=("Arial", 8), bg="white", fg="#999999")
watermark.pack(side=BOTTOM, pady=5)

# Apply a white background to the base
base.config(bg="white")

# Start with placeholder in EntryBox
add_placeholder()

# Start the GUI loop
base.mainloop()
