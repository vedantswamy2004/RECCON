import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to load dataset
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    utterances = []
    emotions = []
    
    for dialogue_id, conversations in data.items():
        for dialogue in conversations:
            for turn in dialogue:
                utterances.append(turn["utterance"])
                emotions.append(turn["emotion"])
    
    return utterances, emotions

# File paths
train_file = "data/original_annotation/dailydialog_train.json"
val_file = "data/original_annotation/dailydialog_valid.json"
test_file = "data/original_annotation/dailydialog_test.json"

# Load datasets
X_train_text, y_train = load_data(train_file)
X_val_text, y_val = load_data(val_file)
X_test_text, y_test = load_data(test_file)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

# Train Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = nb_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = nb_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Report:\n", classification_report(y_test, y_test_pred))

# Function to predict emotions for new sentences
def predict_emotion(text):
    text_vectorized = vectorizer.transform([text])
    return nb_model.predict(text_vectorized)[0]

# Example usage
sample_text = "I feel so excited about this!"
predicted_emotion = predict_emotion(sample_text)
print(f"Predicted Emotion: {predicted_emotion}")