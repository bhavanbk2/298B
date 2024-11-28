import streamlit as st
import os
from dotenv import load_dotenv
import json
import time
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from langchain_openai.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize LangChain's ChatOpenAI client for GPT-3.5
openai_api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Load Hugging Face token
hf_token = os.getenv("Hugging_face_token")

# Function to load Llama 3.2 model and tokenizer
@st.cache_resource
def load_llama_model():
    model_name = "shashikumar1998/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    model.eval()
    return model, tokenizer

# Load Llama 3.2 model and tokenizer
model, tokenizer = load_llama_model()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to generate responses using GPT-3.5
def generate_response_gpt(query):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": f"You are a chatbot impersonating {st.session_state.persona}."},
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

# Function to generate responses using Llama 3.2
def generate_response_llama(query):
    try:
        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Function to simulate typing animation
def typing_animation():
    with st.spinner('Bot is typing...'):
        time.sleep(1)

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox("Select Model", ["GPT-3.5", "Llama 3.2 (Hugging Face)"])
    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Apply theme-specific CSS
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

# Main app
st.title("💬 Persona-Based Conversational Bot")
user_query = st.text_input("Your Question", placeholder="💡 Ask me anything!")

if st.button("Submit"):
    if user_query:
        if model_choice == "GPT-3.5":
            response = generate_response_gpt(user_query)
        elif model_choice == "Llama 3.2 (Hugging Face)":
            response = generate_response_llama(user_query)
        else:
            response = "Model choice not implemented."

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append({"user": user_query, "bot": response})

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
