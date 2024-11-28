import streamlit as st
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_openai.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Explicitly retrieve API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("Hugging_face_token")

# Initialize OpenAI GPT-3.5 Client
gpt_client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Function to load the Llama base model
@st.cache_resource
def load_llama_base_model():
    try:
        model_name = "meta-llama/Llama-2-7b-hf"  # Base Llama model
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.eval()  # Set the model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Llama model: {e}")
        raise

# Load the Llama model and tokenizer
try:
    llama_model, llama_tokenizer = load_llama_base_model()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Function to generate responses using GPT-3.5
def generate_response_gpt(query):
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.get('chat_history', [])])
    messages = [
        {"role": "system", "content": f"You are a chatbot impersonating {st.session_state.get('persona', 'Assistant')}."},
        {"role": "user", "content": f"{context} {query}"}
    ]
    try:
        response = gpt_client(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        st.session_state.chat_history.append({'user': query, 'bot': response_text})
        return response_text
    except Exception as e:
        st.error(f"Error generating response with GPT-3.5: {e}")
        return "Sorry, I couldn't generate a response."

# Function to generate responses using Llama
def generate_response_llama(query):
    try:
        inputs = llama_tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs["input_ids"],
                max_length=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.session_state.chat_history.append({'user': query, 'bot': response})
        return response
    except Exception as e:
        st.error(f"Error generating response with Llama: {e}")
        return "Sorry, I couldn't generate a response."

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant"])
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox("Select Model", ["GPT-3.5", "Llama (Base Model)"])
    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

# Add custom CSS for styling
def apply_custom_css(theme):
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"
    st.markdown(f"""
    <style>
    body {{
        background-color: {primary_color};
        color: {text_color};
    }}
    .stApp {{
        background-color: {primary_color};
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css(theme)

# Main app content
st.title("💬 Persona-Based Conversational Bot")
user_query = st.text_input("Your Question", placeholder="💡 Ask me anything!")

if st.button("Generate"):
    if user_query:
        if model_choice == "GPT-3.5":
            response = generate_response_gpt(user_query)
        elif model_choice == "Llama (Base Model)":
            response = generate_response_llama(user_query)
        else:
            response = "Model choice not implemented."

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
else:
    st.markdown("No conversation yet. Start by asking a question!")
