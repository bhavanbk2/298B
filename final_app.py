import streamlit as st
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Hugging Face Authentication
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    login(HUGGINGFACE_TOKEN)
else:
    st.error("Hugging Face token not found in environment variables. Add it to your .env file.")
    st.stop()

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and model selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history = []

    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox(
        "Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"]
    )
    
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox(
        "Select Model", ["Llama 3.2 (Hugging Face)", "GPT-3.5"]
    )

    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Apply custom CSS based on theme
def apply_custom_css(theme):
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {primary_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {primary_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_custom_css(theme)

# Title and Subtitle
st.markdown("<h1>💬 Persona-based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown("<p>Ask me anything about health and wellness!</p>", unsafe_allow_html=True)

# Chat history session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to generate responses from Llama 3.2 model
def generate_response_llama(query):
    try:
        model_name = "shashikumar1998/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        model.eval()
        inputs = tokenizer(query, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error: {e}"

# Placeholder for user input and display
user_query = st.text_input("Your Question", placeholder="💡 What’s on your mind?", key="user_input")
if st.button("→", key="submit_button"):
    if user_query:
        if model_choice == "Llama 3.2 (Hugging Face)":
            response = generate_response_llama(user_query)
        elif model_choice == "GPT-3.5":
            response = "GPT-3.5 integration is not implemented in this example."
        else:
            response = "Invalid model choice."

        st.session_state.chat_history.append({"user": user_query, "bot": response})

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

