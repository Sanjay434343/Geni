import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents with enhanced error handling
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Load the intents
intents = load_json_file('intents.json')

words = []
classes = []
documents = []

# Process intents
for intent in intents.get('intents', []):
    try:
        for pattern in intent.get('text', []):
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['intent']))
            if intent['intent'] not in classes:
                classes.append(intent['intent'])
    except KeyError as e:
        print(f"Missing key in intent: {e}")
        continue

# Clean up and prepare data
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!']]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Prepare training data
training_sentences = []
training_labels = []

for doc in documents:
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    training_sentences.append(bag)
    training_labels.append(classes.index(doc[1]))

training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_sentences, training_labels, epochs=200, batch_size=10, verbose=1)

# Save the model
model.save('chatbot_model.h5')

print("Model trained and saved as 'chatbot_model.h5'")
