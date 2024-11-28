import streamlit as st
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load environment variables
load_dotenv()

# Explicitly retrieve Hugging Face token
hf_token = os.getenv("Hugging_face_token")
if not hf_token:
    st.error("Hugging Face token is missing. Add it to your .env file.")
    st.stop()

# Function to load base model and adapter using PEFT
@st.cache_resource
def load_llama_with_adapter():
    try:
        # Base model and adapter
        base_model_name = "meta-llama/Llama-2-7b-hf"  # Replace with correct base model
        adapter_path = "shashikumar1998/Llama-3.2-3B-Instruct"  # Hugging Face adapter path

        # Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token)

        # Apply the adapter
        model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)
        model.eval()  # Set model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Llama model with adapter: {e}")
        raise

# Load model and tokenizer
try:
    llama_model, llama_tokenizer = load_llama_with_adapter()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Function to generate responses using Llama
def generate_response(query):
    try:
        # Tokenize the input query
        inputs = llama_tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate response
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs["input_ids"],
                max_length=100,
                do_sample=True,
                temperature=0.7,  # Controls randomness
                top_p=0.9,  # Nucleus sampling
            )

        # Decode and return the response
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Streamlit UI setup
st.set_page_config(page_title="Llama 3.2 with Adapter", layout="wide")

# Main app content
st.title("💬 Llama 3.2 Conversational Bot with Adapter")
user_query = st.text_input("Your Question", placeholder="💡 Ask me anything!")

if st.button("Generate"):
    if user_query:
        response = generate_response(user_query)
        st.markdown(f"**Response:** {response}")
