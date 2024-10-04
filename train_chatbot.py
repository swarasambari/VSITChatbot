import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Process intents to build vocabulary and training data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, removing duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Output statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Generate training data
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output array
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Debug: Print lengths
    print(f"Bag length: {len(bag)}, Output row length: {len(output_row)}")
    
    training.append([bag, output_row])

# Shuffle and convert to np.array
random.shuffle(training)

# Debugging output
for i, entry in enumerate(training):
    print(f"Entry {i}: {entry}, Lengths: Bag - {len(entry[0])}, Output - {len(entry[1])}")

# Check if all bags and output rows are consistent
for entry in training:
    if len(entry[0]) != 88 or len(entry[1]) != 9:
        print("Inconsistent entry found:", entry)

# Convert to np.array (ensure all shapes are consistent)
training = np.array(training, dtype=object)

# Create training lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit and save the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created")
