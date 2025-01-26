# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.metrics import confusion_matrix

# Step 2: Load NLP Models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

# Step 3: Aid Request Classifier (Rule-Based)
def classify_aid_request(text):
    categories = {
        "food": ["hunger", "groceries", "meal", "nutrition", "eat"],
        "medical": ["hospital", "doctor", "medication", "emergency", "sick"],
        "financial": ["money", "fund", "donation", "assistance", "rent"],
        "shelter": ["house", "homeless", "evacuate", "residence", "fire"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in text.lower() for keyword in keywords):
            return category
    return "other"

# Step 4: Sentiment Analysis (Urgency Detection)
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

# Step 5: Create Test Data (Manually Labeled)
test_data = [
    ("I need food for my children.", "food"),
    ("We require financial help for medical bills.", "financial"),
    ("Emergency! My mother needs a doctor immediately.", "medical"),
    ("Our house was destroyed by a typhoon.", "shelter"),
    ("I'm feeling very sick but have no money for medicine.", "medical"),
    ("We are hungry and have nothing to eat.", "food"),
    ("Can anyone help with rent assistance?", "financial"),
    ("There is no place to stay after the fire.", "shelter")
]

df_test = pd.DataFrame(test_data, columns=["text", "actual_category"])

# Step 6: Run Predictions
df_test["predicted_category"] = df_test["text"].apply(classify_aid_request)

# Step 7: Generate Confusion Matrix
category_mapping = {"food": 0, "financial": 1, "medical": 2, "shelter": 3, "other": 4}
df_test["actual_category"] = df_test["actual_category"].map(category_mapping)
df_test["predicted_category"] = df_test["predicted_category"].map(category_mapping)

cm = confusion_matrix(df_test["actual_category"], df_test["predicted_category"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=category_mapping.keys(), yticklabels=category_mapping.keys())
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Aid Request Classification")
plt.show()

# Step 8: Sentiment Analysis & Score Distribution
df_test["sentiment"], df_test["sentiment_score"] = zip(*df_test["text"].apply(analyze_sentiment))

plt.figure(figsize=(6, 4))
sns.histplot(df_test["sentiment_score"], bins=10, kde=True, color="blue")
plt.xlabel("Sentiment Score (Urgency Level)")
plt.ylabel("Frequency")
plt.title("Sentiment Score Distribution")
plt.show()

# Step 9: Print Test Results
print(df_test[["text", "actual_category", "predicted_category", "sentiment", "sentiment_score"]])
