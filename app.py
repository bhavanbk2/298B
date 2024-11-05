import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_openai.chat_models import ChatOpenAI
import embedding as emb

# Load environment variables
load_dotenv()

# Initialize Cohere client
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# Initialize LangChain's ChatOpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Get the embedding index
index = emb.get_index("cohere-pinecone-tree")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Store all queries and responses

# Function to generate responses
def generate_response(query):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": "You are a chatbot impersonating Dr. Sanjay Gupta."},
        {"role": "user", "content": f"{context} {query}"}
    ]

    try:
        response = client(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        st.session_state.chat_history.append({'user': query, 'bot': response_text})
        return response_text

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")
st.title("💬 Persona based Conversational Bot")

# Add custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
    }
    .chat-bubble-user {
        background-color: #DCF8C6;
        align-self: flex-end;
        color: black;
    }
    .chat-bubble-bot {
        background-color: #F1F0F0;
        align-self: flex-start;
        color: black;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# User input for query
st.markdown("### Ask me anything about health and wellness!")

# Create columns for input and buttons
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Your Query:", placeholder="Type your question here...", key="user_input")

with col2:
    if st.button("New Chat"):
        st.session_state.chat_history.clear()  # Clear history for new chat
        if 'user_input' in st.session_state:
            del st.session_state.user_input  # Remove the key to clear input
    if st.button("Clear Chat"):
        st.session_state.chat_history.clear()  # Clear chat history
        if 'user_input' in st.session_state:
            del st.session_state.user_input  # Remove the key to clear input

# Process the user query when "Submit" button is pressed
if st.button("Submit") and user_query:
    response = generate_response(user_query)  # Generate response
    # Clear input field after submission
    if 'user_input' in st.session_state:
        del st.session_state.user_input  # Remove the key to clear input

# Display chat messages in a conversational style
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for interaction in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble chat-bubble-user'><strong>User:</strong> {interaction['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble chat-bubble-bot'><strong>Bot:</strong> {interaction['bot']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("No conversation yet.")
