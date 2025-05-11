import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# Download NLTK data if not present
nltk.download('punkt')
nltk.download('wordnet')

# Load intents data
with open('festivals_data.json') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Load the trained model (make sure to save your model as 'festivals_chatbot_model.h5' from the notebook)
model_path = 'festivals_chatbot_model.h5'
if not os.path.exists(model_path):
    print('Model file not found! Please run and save the model in the notebook as festivals_chatbot_model.h5.')
    exit(1)
model = load_model(model_path)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent'] if ints else None
    if tag:
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return np.random.choice(i['responses'])
    return "Sorry, I do not understand."

def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, data)
    return res

if __name__ == "__main__":
    print("Indian Festivals Chatbot. Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(inp)
        print("Chatbot:", response)
