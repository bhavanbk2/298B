import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import streamlit as st
import time
import json

# Load LLaMA model and tokenizer from the local path
llama_model_dir = r'C:\Users\Checkout\Downloads\llama3_2_3b' # Use raw string for Windows paths

# Load the LLaMA model and tokenizer
llama_model = LlamaForCausalLM.from_pretrained(llama_model_dir)
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_dir)

# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to generate LLaMA model response
def generate_llama_response(query):
    inputs = llama_tokenizer(query, return_tensors="pt")

    with torch.no_grad():
        outputs = llama_model.generate(inputs['input_ids'], max_length=200, num_return_sequences=1)

    response_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

# Simulate typing animation
def typing_animation():
    with st.spinner('Bot is typing...'):
        time.sleep(1)

# Function to clear chat history
def clear_chat():
    st.session_state.chat_history.clear()

# Streamlit UI setup
st.set_page_config(page_title="Persona Based Conversational Bot", layout="wide")

# Sidebar for model and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        clear_chat()

    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])

    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Apply custom CSS based on theme
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
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS based on theme
apply_custom_css(theme)

# Display the title and subtitle
st.markdown(f"<h1 class='title-text'>💬 Persona based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>Ask me anything about health and wellness!</p>", unsafe_allow_html=True)

# Select the model to use from a dropdown (LLaMA 3 and GPT-3 only)
selected_bot = st.selectbox("Choose your chatbot model", ["LLaMA 3", "GPT-3"])

# Create input box for user query
user_query = st.text_input("", placeholder="💡 What’s on your mind?", key="user_input", label_visibility="collapsed")

# Generate response based on selected model
if st.button("→", key="submit_button", help="Submit your query"):
    if user_query:
        typing_animation()  # Simulate typing
        if selected_bot == "LLaMA 3":
            response = generate_llama_response(user_query)
        elif selected_bot == "GPT-3":
            response = "GPT-3 response placeholder (integrate OpenAI API here)"
        
        # Store the conversation history
        st.session_state.chat_history.append({'user': user_query, 'bot': response})

# Clear chat button
if st.button("Clear Chat"):
    clear_chat()

# Display chat messages
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for interaction in st.session_state.chat_history:
        st.markdown(f"""
            <div class='chat-bubble chat-bubble-user'>
                <strong>User:</strong> {interaction['user']}
            </div>
            <div class='chat-bubble chat-bubble-bot'>
                <strong>Bot:</strong> {interaction['bot']}
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<p class='no-conversation'>🤖 No conversations yet. Ask a question!</p>", unsafe_allow_html=True)

# Add option to download chat history
if st.button("Download Chat History"):
    chat_history_json = json.dumps(st.session_state.chat_history, indent=4)
    st.download_button(label="Download", data=chat_history_json, file_name="chat_history.json", mime="application/json")
