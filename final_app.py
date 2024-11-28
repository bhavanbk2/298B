import streamlit as st
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Explicitly retrieve Hugging Face token
hf_token = os.getenv("Hugging_face_token")
if not hf_token:
    st.error("Hugging Face token is missing. Add it to your .env file.")
    st.stop()

# Function to load the Llama 3.2 model and tokenizer
@st.cache_resource
def load_llama_model():
    try:
        model_name = "shashikumar1998/Llama-3.2-3B-Instruct"  # Your Hugging Face model
        # Load tokenizer and model using the Hugging Face token
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        model.eval()  # Set the model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Llama 3.2 model: {e}")
        raise

# Load Llama 3.2 model and tokenizer
try:
    llama_model, llama_tokenizer = load_llama_model()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Function to generate responses using Llama 3.2
def generate_response_llama(query):
    try:
        # Tokenize the input query
        inputs = llama_tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate response
        with torch.no_grad():
            outputs = llama_model.generate(inputs["input_ids"], max_length=100, do_sample=True)

        # Decode and return the response
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Streamlit UI setup
st.set_page_config(page_title="Llama 3.2 Bot", layout="wide")

# Main app content
st.title("💬 Llama 3.2 Conversational Bot")
user_query = st.text_input("Your Question", placeholder="💡 Ask me anything!")

if st.button("Generate"):
    if user_query:
        response = generate_response_llama(user_query)
        st.markdown(f"**Response:** {response}")
