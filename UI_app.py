import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_openai.chat_models import ChatOpenAI
import embedding as emb
import json
import time
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

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
    st.session_state.chat_history = []

# Function to handle authentication with Hugging Face
def authenticate_hugging_face(token):
    login(token)  # This will authenticate using the provided token

# Function to generate responses using GPT-3.5
def generate_response_gpt(query):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": f"You are a chatbot impersonating {st.session_state.persona}."},
        {"role": "user", "content": f"{context} {query}"}
    ]
    typing_animation()  # Simulate typing

    try:
        response = client(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        st.session_state.chat_history.append({'user': query, 'bot': response_text})
        return response_text

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Function to generate responses from local Llama 3.2 model
def generate_response_llama(query):
    try:
        # Local path to your Llama model folder
        model_path = r"C:\Users\Checkout\Downloads\llama3_2_3b"  # Update this to your actual path
        tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, force_download=False)

        # Ensure the model is in evaluation mode
        model.eval()

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt")

        # Generate response (with no gradients)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error generating response: {e}")  # Log error for debugging
        return f"Error: {e}"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
        
# Function to simulate typing animation
def typing_animation():
    with st.spinner('Bot is typing...'):
        time.sleep(1)  # Simulate typing delay

# Load images from the repository
def load_image(image_path):
    if os.path.exists(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode()
    return None

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox("Select Model", ["GPT-3.5", "Llama 3.2 (Local)", "Other Models"])

    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Load images from repository
user_avatar_path = "images/user_image.png"
bot_avatar_path = "images/bot_image.png"

user_avatar_base64 = load_image(user_avatar_path)
bot_avatar_base64 = load_image(bot_avatar_path)

# Ensure avatars are available
if user_avatar_base64 is None or bot_avatar_base64 is None:
    st.error("User and/or Bot avatars are missing. Please ensure the images are in the 'images' folder.")
    st.stop()

# Add custom CSS for styling based on theme
def apply_custom_css(theme):
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"
    question_mark_color = "#6C757D"  # Softer color for the question mark

    st.markdown(f"""
    <style>
    body {{
        background-color: {primary_color};
        color: {text_color};
        font-family: 'Arial', sans-serif;
    }}
    .stApp {{
        background-color: {primary_color};
    }}
    .chat-bubble {{
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
    }}
    .chat-bubble-user {{
        background-color: #DCF8C6;
        align-self: flex-end;
        color: black;
    }}
    .chat-bubble-bot {{
        background-color: #F1F0F0;
        align-self: flex-start;
        color: black;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }}
    .avatar {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }}
    .title-text {{
        font-size: 36px; /* Increased size */
        font-weight: bold;
        color: {text_color};
    }}
    .subtitle-text {{
        font-size: 20px; /* Increased size */
        color: {text_color};
        margin-top: -10px;
    }}
    .no-conversation {{
        font-size: 20px;
        color: #ffc107;
        text-align: center;
        margin-top: 20px;
        font-style: italic;
    }}
    .tooltip {{
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: pointer;
        margin-left: 5px;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 150px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        left: 50%; /* Center the tooltip */
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px; /* Smaller font for the tooltip */
        bottom: 30px; /* Adjusted position */
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    .question-mark {{
        width: 20px; /* Smaller size */
        height: 20px; /* Smaller size */
        border-radius: 50%;
        background-color: {question_mark_color};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px; /* Slightly smaller font */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS based on theme
apply_custom_css(theme)

# Display the title and subtitle
st.markdown(f"<h1 class='title-text'>💬 Persona based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>Ask me anything about health and wellness!</p>", unsafe_allow_html=True)

# Tooltip with a refined question mark icon
st.markdown("""<div style='display: flex; align-items: center; margin-bottom: 10px;'>
    <div class='tooltip'>
        <div class='question-mark'>?</div>
        <span class='tooltiptext'>Enter your question below</span>
    </div>
</div>""", unsafe_allow_html=True)

# Create a container for input and submission
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input("Your Question", placeholder="💡 What’s on your mind?", key="user_input", label_visibility="collapsed")

with col2:
    if st.button("→", key="submit_button", help="Submit your query"):
        if user_query:
            if model_choice == "GPT-3.5":
                response = generate_response_gpt(user_query)
            elif model_choice == "Llama 3.2 (Local)":
                response = generate_response_llama(user_query)
            else:
                response = "Model choice not implemented."

            if 'user_input' in st.session_state:
                del st.session_state.user_input  # Remove the key to clear input

# Display chat messages in a conversational style
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for interaction in st.session_state.chat_history:
        st.markdown(f"""
            <div class='chat-bubble chat-bubble-user'>
                <img class='avatar' src='data:image/png;base64,{user_avatar_base64}' alt='User Avatar'/>
                <strong>User:</strong> {interaction['user']}
            </div>
            <div class='chat-bubble chat-bubble-bot'>
                <img class='avatar' src='data:image/png;base64,{bot_avatar_base64}' alt='Bot Avatar'/>
                <strong>Bot:</strong> {interaction['bot']}
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<p class='no-conversation'>🤖 No conversations yet. Ask a question!</p>", unsafe_allow_html=True)
