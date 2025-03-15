import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Using SVM for better accuracy

# SSL Fix for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('stopwords')
nltk.download('wordnet')

# Load intents from JSON file
file_path = os.path.abspath("C:/Users/vishn/OneDrive/Desktop/Chatbot/intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text: Lowercase + Stopword Removal (NO tokenization)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split by spaces (NO tokenization)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords
    return " ".join(filtered_words)

# Function to get synonyms from WordNet (LIMITED to 2 synonyms per word)
def get_synonyms(word, max_synonyms=2):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores with spaces
            if len(synonyms) >= max_synonyms:
                return synonyms
    return synonyms

# Function to expand patterns using synonyms (Ensuring tag alignment)
def expand_patterns(patterns, tags):
    expanded_patterns = []
    expanded_tags = []
    for pattern, tag in zip(patterns, tags):
        words = pattern.split()
        expanded = set(words)  # Ensure uniqueness
        for word in words:
            expanded.update(get_synonyms(word))  # Add up to 2 synonyms per word
        expanded_patterns.append(" ".join(expanded))
        expanded_tags.append(tag)  # Maintain tag alignment
    return expanded_patterns, expanded_tags

# Extract patterns and tags
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(preprocess_text(pattern))  # Preprocess patterns

# Expand patterns using synonyms while keeping alignment with tags
patterns, tags = expand_patterns(patterns, tags)

# Feature extraction using TF-IDF (Unigrams + Bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
x = vectorizer.fit_transform(patterns)
y = tags

# Train SVM classifier (Better accuracy than Logistic Regression)
clf = SVC(kernel='linear', probability=True)
clf.fit(x, y)

# Chatbot function with confidence threshold
def chatbot(input_text):
    processed_text = preprocess_text(input_text)  # Preprocess user input
    print(f"Processed Input: {processed_text}")  # Debugging

    input_vector = vectorizer.transform([processed_text])
    probs = clf.predict_proba(input_vector)[0]  # Get probability scores
    max_prob = max(probs) if probs.size > 0 else 0  # Ensure probs is not empty

    print(f"Prediction Probabilities: {probs}")  # Debugging
    print(f"Max Probability: {max_prob}")  # Debugging

    # Confidence-based response selection
    if max_prob < 0.5:
        return "I'm not sure how to respond. Can you rephrase?"

    tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Initialize counter
counter = 0

# Streamlit UI for chatbot
def main():
    global counter
    st.title("Intents-Based Chatbot Using NLP & SVM")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message and press Enter to start the conversation.")

        # Ensure chat log file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Log conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("This chatbot uses NLP, TF-IDF, and SVM to predict responses based on predefined intents.")

        st.subheader("Project Enhancements:")
        st.write("""
        - **Text Preprocessing**: Stopword removal (No tokenization).
        - **TF-IDF with Bigrams**: Captures context better than unigrams.
        - **SVM Model**: Improved classification over Logistic Regression.
        - **Confidence-Based Responses**: Avoids incorrect replies.
        - **Synonym Expansion**: Improves generalization of training patterns.
        """)

if __name__ == '__main__':
    main()
