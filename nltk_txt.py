import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Function to download NLTK data if not already present
def download_nltk_data():
    nltk_data_needed = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
    for data in nltk_data_needed:
        nltk.download(data, quiet=True)
    nltk.download('omw-1.4')

# Streamlit app title
st.title('NLTK Text Summarizer')

# Downloading NLTK data only if necessary
download_nltk_data()

# Text input
user_input_text = st.text_area("Enter Text", "Natural language processing enables computers to understand human language. This technology is behind voice-activated assistants, online customer support, and more.", height=300)

# Selection for the number of sentences in the summary
num_sentences = st.slider('Number of sentences for summary:', min_value=1, max_value=10, value=2)

def summarize_text(text, num_sentences):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]

    word_scores = {}
    for word in filtered_words:
        synsets = wn.synsets(word)
        if synsets:
            word_scores[word] = len(synsets)

    sentence_scores = {}
    for sentence in sent_tokenize(text):
        sentence_scores[sentence] = sum(word_scores.get(word, 0) for word in word_tokenize(sentence))

    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(sorted_sentences[:num_sentences])
    return summary

if st.button('Summarize'):
    summary_result = summarize_text(user_input_text, num_sentences)
    st.write(summary_result)
