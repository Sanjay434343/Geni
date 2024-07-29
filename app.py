import re
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random
import requests

app = Flask(__name__)

# Load the model and necessary data
model = tf.keras.models.load_model('chatbot_model.h5')

lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['text']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['intent']))
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!']]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# List of trademarked phrases or terms
TRADEMARKED_PHRASES = [
    "your trademarked phrase",
    "another trademarked term",
    "example trademarked term"
]

# Special symbols to ignore or replace
SYMBOLS_TO_IGNORE = ['®', '™']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/spk')
def speak():
    return render_template('spk.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    
    # Clean message and check for trademarked phrases
    cleaned_message = clean_up_message(message)
    if contains_trademarked_phrases(cleaned_message):
        return jsonify({"text": "Your message contains trademarked terms and cannot be processed."})
    
    ints = predict_class(cleaned_message)
    response = get_response(ints, intents)
    
    # Check if the response requires weather info
    if "WeatherQuery" in [intent['intent'] for intent in ints]:
        city = extract_city(message)
        weather_response = get_weather(city)
        response = weather_response
    
    # Clean response text
    response["text"] = clean_up_message(response["text"])
    
    return jsonify(response)

def contains_trademarked_phrases(message):
    # Convert message to lowercase for case-insensitive comparison
    message = message.lower()
    return any(phrase in message for phrase in TRADEMARKED_PHRASES)

def clean_up_message(message):
    # Remove specific special characters and symbols
    message = re.sub(r'[\'":;,.]', '', message)
    for symbol in SYMBOLS_TO_IGNORE:
        message = message.replace(symbol, '')
    return message

def clean_up_sentence(sentence):
    sentence = re.sub(r'[\'":;,.]', '', sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
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
        if i['intent'] == tag:
            result = random.choice(i['responses'])
            break
    return {"text": result}

def extract_city(message):
    # Placeholder function for extracting city name
    return "pollachi"  # Example static city

def get_weather(city):
    api_key = '97ed86b99fdcf738c7a080e0fa9fde20'
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data['cod'] == 200:
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return {
            "text": f"The current weather in {city} is {weather_description} with a temperature of {temperature}°C."
        }
    else:
        return {
            "text": f"Sorry, I couldn't retrieve the weather information for {city}."
        }

if __name__ == '__main__':
    app.run(debug=True)
