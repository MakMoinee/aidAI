import logging
from transformers import pipeline
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, origins=["http://localhost:8443"])

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

# Load Named Entity Recognition (NER) Model
nlp = spacy.load("en_core_web_sm")

def classify_aid_request(text):
    categories = {
        "food": ["hunger", "groceries", "meal", "nutrition"],
        "medical": ["hospital", "doctor", "medication", "emergency", "treatment"],
        "financial": ["money", "fund", "donation", "assistance", "education"],
        "shelter": ["house", "homeless", "evacuate", "residence"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in text.lower() for keyword in keywords):
            logging.info(f"Classified as: {category} (Matched keyword in {keywords})")
            return category
    
    logging.info("Classified as: other (No matching keywords)")
    return "other"

def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    logging.info(f"Extracted entities: {entities}")
    return entities

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    logging.info(f"Sentiment Analysis Result: {result}")
    return result['label'], result['score']

@app.route('/process_request', methods=['POST'])
def process_request():
    data = request.json
    text = data.get("message", "")

    logging.info(f"Received request: {text}")

    # Perform NLP tasks
    category = classify_aid_request(text)
    entities = extract_entities(text)
    sentiment, score = analyze_sentiment(text)

    response = {
        "category": category,
        "entities": entities,
        "urgency_score": score
    }

    logging.info(f"Response: {response}")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
