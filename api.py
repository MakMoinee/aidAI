# Step 1: Install required libraries before running this script
# pip install transformers torch spacy nltk

from transformers import pipeline
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8443"])

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

# Load Named Entity Recognition (NER) Model
nlp = spacy.load("en_core_web_sm")

# Step 2: Function to classify aid requests
def classify_aid_request(text):
    categories = {
        "food": ["hunger", "groceries", "meal", "nutrition"],
        "medical": ["hospital", "doctor", "medication", "emergency","treatment"],
        "financial": ["money", "fund", "donation", "assistance", "education"],
        "shelter": ["house", "homeless", "evacuate", "residence"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in text.lower() for keyword in keywords):
            return category
    return "other"

# Step 3: Named Entity Recognition (Extracting entities)
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Step 3: Sentiment Analysis (Detect urgency)
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']


@app.route('/process_request', methods=['POST'])
def process_request():
    data = request.json
    text = data.get("message", "")

    # Perform NLP tasks
    category = classify_aid_request(text)
    entities = extract_entities(text)
    sentiment, score = analyze_sentiment(text)

    response = {
        "category": category,
        "entities": entities,
        "sentiment": sentiment,
        "urgency_score": score
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# # Example usage
# if __name__ == "__main__":
#     request_text = "I am Kennen from Manila. My family needs money because we have been hit with fire"

#     # Classify request
#     category = classify_aid_request(request_text)

#     # Extract entities
#     entities = extract_entities(request_text)

#     # Analyze sentiment
#     sentiment, score = analyze_sentiment(request_text)

#     # Print results
#     print(f"Category: {category}")
#     print(f"Entities: {entities}")
#     print(f"Sentiment: {sentiment}, Urgency Score: {score}")
