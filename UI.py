import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_openai.chat_models import ChatOpenAI
import embedding as emb
from textblob import TextBlob
import json
import time
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

# Function to generate responses for Cohere or OpenAI models
def generate_response(query, model="openai"):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": f"You are a chatbot impersonating {st.session_state.persona}."},
        {"role": "user", "content": f"{context} {query}"}
    ]
    typing_animation()  # Simulate typing

    try:
        if model == "openai":
            response = client(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        elif model == "cohere":
            response = co.generate(prompt=query, max_tokens=200)
            response_text = response.text
        elif model == "llama":
            response_text = generate_llama_response(query)
        st.session_state.chat_history.append({'user': query, 'bot': response_text})
        return response_text

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Function to generate responses for Llama-3.2 model
def generate_llama_response(query):
    # Load the model and tokenizer
    model_name = "shashikumar1998/Llama-3.2-3B-Instruct"
    config = AutoConfig.from_pretrained(model_name)
    
    # Modify the rope_scaling dictionary to fit the expected format
    config.rope_scaling = {
        'type': 'fixed',  # or another valid type
        'factor': 32.0
    }

    # Load the model with the modified config
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input and generate a response
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

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
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)
    st.markdown("### 🧑‍💻 Choose Model")
    model_choice = st.selectbox("Select Model", ["OpenAI GPT-3.5", "Cohere", "Llama-3.2"])

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
    .question-mark {{
        color: {question_mark_color};
        font-size: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css(theme)

# Chat container with message inputs and avatar images
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("", label_visibility="collapsed")  # Hidden label
with col2:
    submit_button = st.button("Send")

# Handle input and generate bot response
if submit_button and user_query:
    st.session_state.chat_history.append({'user': user_query, 'bot': "..."})  # Placeholder
    response = generate_response(user_query, model=model_choice)

    # Display the conversation with avatars
    with st.container():
        chat_container = st.empty()
        chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message["user"]:
                chat_container.markdown(f'<div class="chat-bubble chat-bubble-user"><img class="avatar" src="data:image/png;base64,{user_avatar_base64}"> {message["user"]}</div>', unsafe_allow_html=True)
            if message["bot"]:
                chat_container.markdown(f'<div class="chat-bubble chat-bubble-bot"><img class="avatar" src="data:image/png;base64,{bot_avatar_base64}"> {message["bot"]}</div>', unsafe_allow_html=True)
        
        chat_container.markdown('</div>', unsafe_allow_html=True)

    # Rerun to update chat display
    st.rerun()
