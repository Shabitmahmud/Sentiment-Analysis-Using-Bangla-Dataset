# app.py
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\akash\sentiment_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        input_text = request.form['input_text']

        # Split the input paragraph into sentences using "।" and "?"
        sentences = re.split(r'[।?]', input_text)

        results = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Tokenize and analyze sentiment for each sentence
            tokenized_input = tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors="pt")
            output = model(**tokenized_input, return_dict=True)

            # Get the sentiment label and confidence score
            predicted_label = output.logits.argmax().item()
            sentiment_mapping = {0: "negative", 1: "positive"}
            sentiment = sentiment_mapping[predicted_label]
            probabilities = torch.softmax(output.logits, dim=1)
            positive_percentage = probabilities[0, 1].item() * 100  # Percentage for positive class
            negative_percentage = probabilities[0, 0].item() * 100

            results.append({
              'sentence': sentence,
              'sentiment': sentiment,
              'positive_percentage': positive_percentage,
              'negative_percentage': negative_percentage
            })

        return jsonify(results)

if __name__ == '__main__':
    app.run()
