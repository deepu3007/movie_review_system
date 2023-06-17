import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import nltk
import ssl
import streamlit as st
from transformers import pipeline
from translate import Translator
import certifi

# Set the SSL certificate path
nltk.data.path.append(certifi.where())

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Rest of your code

# Function to preprocess the review text
def preprocess_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token not in punctuation]
    
    # Join the processed tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Load the dataset
df = pd.read_csv('/Users/deepesh/Desktop/Movie_Review_System/IMDB_Dataset.csv')

# Preprocess the review text
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

df['processed_review'] = df['review'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = df['processed_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a LinearSVC model
model = LinearSVC()
model.fit(X_train_vectors, y_train)

# Save the model
with open('sentiment_analysis_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the sentiment analysis model
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the summarization model
summarizer = pipeline('summarization')

# Function to generate sentiment
def generate_sentiment(review):
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    sentiment = model.predict(review_vector)[0]
    return sentiment

# Function to generate summary
def generate_summary(review):
    summary = summarizer(review, max_length=150, min_length=4, do_sample=False)
    return summary[0]['summary_text']

# Function to translate summary
def translate_summary(summary, target_lang):
    translator = Translator(to_lang=target_lang)
    translation = translator.translate(summary)
    return translation

# Main function for Streamlit app
def main():
    # Set page title and custom CSS styling
    st.set_page_config(page_title="Movie Review System", page_icon="🎥")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F5F5F5;
        }
        .stTextInput, .stButton {
            background-color: #FFFFFF;
        }
        .stTextArea textarea {
            background-color: #FFFFFF;
        }
        .stHeader {
            color: #FF9800;
            font-size: 36px;
            text-align: center;
        }
        .stSubheader {
            color: #FF9800;
            font-size: 24px;
            margin-top: 30px;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Get user input
    st.title("Movie Review System")
    user_review = st.text_area("Enter your movie review here:")

    # Perform sentiment analysis
    sentiment = generate_sentiment(user_review)
    sentiment_color = "positive" if sentiment == "positive" else "negative"
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment: <span class='{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)

    # Generate summary
    summary = generate_summary(user_review)
    st.subheader("Text Summarization")
    st.write(f"Summary: {summary}")

    # Translate summary
    target_lang_mapping = {
        'Hindi': 'hi',
        'Telugu': 'te',
        'Tamil': 'ta',
        'Kannada': 'kn',
        'Bengali': 'bn',
        'Gujarati': 'gu',
        'Malayalam': 'ml',
        'Marathi': 'mr',
        'Punjabi': 'pa'
    }
    target_lang = st.selectbox("Select target language for translation:", list(target_lang_mapping.keys()))
    target_lang_code = target_lang_mapping.get(target_lang)
    if target_lang_code:
        translation = translate_summary(summary, target_lang_code)
        st.subheader("Translation")
        st.write(f"Translated Summary: {translation}")

# Run the app
if __name__ == '__main__':
    main()
