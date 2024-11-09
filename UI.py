import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import embedding as emb
from textblob import TextBlob
import json
import time
import base64

# Set the page configuration first to prevent errors
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Load environment variables
load_dotenv()

# Initialize Cohere client
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# Initialize LangChain's ChatOpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Llama model name on Hugging Face
llama_model_name = "shashikumar1998/Llama-3.2-3B-Instruct"

# Attempt to load Llama model and tokenizer
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
    llama_available = True
except Exception as e:
    st.error(f"Failed to load Llama model: {e}")
    llama_available = False

# Get the embedding index
index = emb.get_index("cohere-pinecone-tree")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to generate responses based on selected model
def generate_response(query, model_choice):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": f"You are a chatbot impersonating {st.session_state.persona}."},
        {"role": "user", "content": f"{context} {query}"}
    ]
    typing_animation()  # Simulate typing

    try:
        if model_choice == "Cohere":
            response = co.generate(
                model="xlarge",
                prompt=query,
                max_tokens=100
            ).generations[0].text.strip()
        elif model_choice == "OpenAI":
            response = client(messages).content if hasattr(client(messages), 'content') else str(client(messages))
        elif model_choice == "Llama" and llama_available:
            inputs = llama_tokenizer(query, return_tensors="pt")
            outputs = llama_model.generate(**inputs, max_new_tokens=50)
            response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = "Selected model is unavailable."

        st.session_state.chat_history.append({'user': query, 'bot': response})
        return response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Add a typing animation to simulate response generation
def typing_animation():
    with st.spinner('Bot is typing...'):
        time.sleep(1)  # Simulate typing delay

# Load images from the repository
def load_image(image_path):
    if os.path.exists(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode()
    return None

# Streamlit UI setup
# Sidebar for persona, theme, and model selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)
    st.markdown("### 🤖 Select Model")
    model_choice = st.selectbox("Select Model", ["Cohere", "OpenAI", "Llama"])

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
    question_mark_color = "#6C757D"

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
        font-size: 36px;
        font-weight: bold;
        color: {text_color};
    }}
    .subtitle-text {{
        font-size: 20px;
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
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        bottom: 30px;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    .question-mark {{
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: {question_mark_color};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS based on theme
apply_custom_css(theme)

# Display title and subtitle
st.markdown("<h1 class='title-text'>💬 Persona based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Ask me anything about health and wellness!</p>", unsafe_allow_html=True)

# Input field for the user’s question
user_query = st.text_input("", placeholder="💡 What’s on your mind?", key="user_input")

# Generate and display response if user enters a query
if user_query:
    response = generate_response(user_query, model_choice)
    st.session_state.chat_history.append({'user': user_query, 'bot': response})
    del st.session_state.user_input  # Clear input after submission

# Display chat messages
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
    st.markdown("<p class='no-conversation'>No conversation history yet!</p>", unsafe_allow_html=True)
