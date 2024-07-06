import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to transform input text
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as v:
    tfidf = pickle.load(v)

with open('model.pkl', 'rb') as m:
    model = pickle.load(m)

# Function to save message history to a file
def save_message_history(history):
    with open("message_history.pkl", "wb") as f:
        pickle.dump(history, f)

# Function to load message history from a file
def load_message_history():
    if os.path.exists("message_history.pkl"):
        with open("message_history.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return []

# Streamlit app
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for better styling with a light background
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f0f0; /* Light gray background */
        color: #333;
    }
    .stTextArea textarea {
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        background-color: white; /* White text area background */
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
    }
    .stAlert p {
        margin: 0;
    }
    .message-history {
        background-color: white; /* White background for message history */
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with information
st.sidebar.title("About the App")
st.sidebar.info(
    """
    This application uses a machine learning model to classify if a given Email/SMS message is **Spam** or **Not Spam**.
    - **Preprocessing**: Tokenization, Stopwords removal, and Stemming.
    - **Model**: Trained on a dataset of spam and ham messages.
    """
)
st.sidebar.markdown(
    """
    ---
    Made by [Rahul Naha](https://github.com/your_github)
    """
)

st.title("📧 Email/SMS Spam Classifier")
st.markdown(
    """
    Enter your message in the text box below to check if it is **Spam** or **Not Spam**.
    """
)

# Initialize message history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = load_message_history()

# Initialize input state
if 'input_sms' not in st.session_state:
    st.session_state['input_sms'] = ""

# Text area for input
input_sms = st.text_area("📝 Enter the message", height=150, value=st.session_state['input_sms'])

if st.button('Predict'):
    if input_sms.strip():  # Check if input is not empty
        with st.spinner('Analyzing...'):
            # Preprocess
            transformed_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]
            # Display prediction
            if result == 1:
                st.error("🚨 This is Spam!")
                st.session_state['message_history'].append((input_sms, "🚨 Spam"))
            else:
                st.success("✅ This is Not Spam!")
                st.session_state['message_history'].append((input_sms, "✅ Not Spam"))
        st.session_state['input_sms'] = ""  # Clear the input field after prediction
        save_message_history(st.session_state['message_history'])  # Save history to file
    else:
        st.warning("⚠️ Please enter a message to classify.")

if st.button('Clear Input'):
    st.session_state['input_sms'] = ""  # Clear input field
    st.session_state['message_history'] = []
    save_message_history(st.session_state['message_history'])  # Save cleared history to file
    st.experimental_rerun()

# Display message history
if st.session_state['message_history']:
    st.subheader("Message History")
    for msg, label in reversed(st.session_state['message_history'][-5:]):
        st.markdown(f"""
        <div class="message-history">
            <p><strong>Message:</strong> {msg}</p>
            <p><strong>Classification:</strong> {label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    """
    ---
    Made  by [Rahul Naha](https://github.com/your_github)
    """
)
