import streamlit as st
import os
from dotenv import load_dotenv
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from langchain_openai.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize LangChain's ChatOpenAI client for GPT-3.5
openai_api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Load Hugging Face token
hf_token = os.getenv("Hugging_face_token")

# Function to load Llama 3.2 base model with LoRA adapter
@st.cache_resource
def load_llama_with_adapter():
    try:
        base_model_name = "meta-llama/Llama-2-7b-hf"  # Compatible base model
        adapter_path = "shashikumar1998/Llama-3.2-3B-Instruct"  # Hugging Face path for adapter

        # Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, use_auth_token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, use_auth_token=hf_token)

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()  # Set model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Llama 3.2 model with LoRA adapter: {e}")
        raise

# Load Llama 3.2 with adapter
try:
    llama_model, llama_tokenizer = load_llama_with_adapter()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

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
        inputs = llama_tokenizer(query, return_tensors="pt").to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = llama_model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode and return the response
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox("Select Model", ["GPT-3.5", "Llama 3.2 (Hugging Face with LoRA)"])
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
        elif model_choice == "Llama 3.2 (Hugging Face with LoRA)":
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
