import json
import nltk
import numpy as np
import random
import tensorflow as tf
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import langid

nltk.download('punkt')

stemmer = LancasterStemmer()

# Load the first model
model1 = tf.keras.models.load_model('EngChatbotModel.h5')

# Load the second model
model2 = tf.keras.models.load_model('InChatBot_Model.h5')

with open('EnglishData.json', 'r') as file:
    data1 = json.load(file)

with open('indataset.json', 'r') as file:
    data2 = json.load(file)

# Extract words and labels for the first model
words1 = []
labels1 = []

for intent in data1['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words1.extend(wrds)
    if intent['tag'] not in labels1:
        labels1.append(intent['tag'])

# Extract words and labels for the second model
words2 = []
labels2 = []

for intent in data2['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words2.extend(wrds)
    if intent['tag'] not in labels2:
        labels2.append(intent['tag'])

# Word Stemming for the first model
words1 = [stemmer.stem(w.lower()) for w in words1 if w != "?"]
words1 = sorted(list(set(words1)))
labels1 = sorted(labels1)

# Word Stemming for the second model
words2 = [stemmer.stem(w.lower()) for w in words2 if w != "?"]
words2 = sorted(list(set(words2)))
labels2 = sorted(labels2)

# Function to convert input to a bag of words
def bag_of_words(s, words, expected_length):
    bag = [0 for _ in range(expected_length)]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se and i < expected_length:
                bag[i] = 1

    return [bag]

# Function to detect language
def detect_language(inp):
    lang, _ = langid.classify(inp)
    return lang

# Function to get a response from the chatbot
def get_response(inp, model1, words1, labels1, expected_length1, model2, words2, labels2, expected_length2):
    lang = detect_language(inp)

    if lang == 'en':
        model = model1
        words = words1
        labels = labels1
        expected_length = expected_length1
    elif lang == 'id':
        model = model2
        words = words2
        labels = labels2
        expected_length = expected_length2
    else:
        return "Unsupported language."

    results = model.predict(np.array(bag_of_words(inp, words, expected_length)))
    results_index = np.argmax(results)
    tag = labels[results_index]

    if lang == 'en':
        data = data1
    elif lang == 'id':
        data = data2

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    return random.choice(responses)

# Main chat loop
print("Start talking with the bot (type quit to stop)!")
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break

    response = get_response(inp, model1, words1, labels1, model1.input_shape[1],
                            model2, words2, labels2, model2.input_shape[1])

    print(response)