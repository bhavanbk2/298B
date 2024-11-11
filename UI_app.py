import streamlit as st
import os
import time
import base64
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle authentication with Hugging Face
def authenticate_hugging_face(token):
    login(token)  # This will authenticate using the provided token

# Function to generate responses using GPT-3.5
def generate_response_gpt(query):
    # Example response logic using GPT-3.5 (adjusted for simplicity)
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [{"role": "system", "content": f"You are a chatbot impersonating {st.session_state.persona}."},
                {"role": "user", "content": f"{context} {query}"}]
    typing_animation()  # Simulate typing
    return "GPT-3.5 response to: " + query  # Placeholder for GPT-3.5 logic

# Function to generate responses from Hugging Face Llama model
def generate_response_llama(query):
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use a stable Llama model for testing
        authenticate_hugging_face("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")  # Login to Hugging Face

        print("Loading model and tokenizer...")  # Debugging statement
        # Load model and tokenizer with forced download
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)

        # Check if model is loaded correctly
        print("Model and tokenizer loaded successfully.")  # Debugging statement

        # Ensure the model is in evaluation mode
        model.eval()

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt")

        print("Query tokenized.")  # Debugging statement

        # Generate response (with no gradients)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        print("Response generated.")  # Debugging statement

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}"

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
    model_choice = st.selectbox("Select Model", ["GPT-3.5", "Llama 2.7b (Hugging Face)", "Other Models"])

    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Load images from repository
user_avatar_path = "images/user_image.png"
bot_avatar_path = "images/bot_image.png"
user_avatar_base64 = load_image(user_avatar_path)
bot_avatar_base64 = load_image(bot_avatar_path)

# Add custom CSS for styling based on theme
def apply_custom_css(theme):
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"
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
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS based on theme
apply_custom_css(theme)

# Display the title and subtitle
st.markdown(f"<h1 class='title-text'>💬 Persona based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>Ask me anything about health and wellness!</p>", unsafe_allow_html=True)

# User input section
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Your Question", placeholder="💡 What’s on your mind?", key="user_input", label_visibility="collapsed")

with col2:
    if st.button("→", key="submit_button", help="Submit your query"):
        if user_query:
            if model_choice == "GPT-3.5":
                response = generate_response_gpt(user_query)
            elif model_choice == "Llama 2.7b (Hugging Face)":
                response = generate_response_llama(user_query)
            else:
                response = "Model choice not implemented."

            # Store the conversation in session state
            st.session_state.chat_history.append({'user': user_query, 'bot': response})

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
