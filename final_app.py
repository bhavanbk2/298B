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

# Function to load the base model and adapter
@st.cache_resource
def load_llama_with_adapter():
    try:
        base_model_name = "./Llama-3.2-3B-Instruct"  # Local path for faster loading
        adapter_path = "./Llama-3.2-3B-Instruct"

        # Load the tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token)

        # Load the adapter
        model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)
        model.eval()  # Set the model to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load Llama model with adapter: {e}")
        raise

# Load the model and tokenizer
try:
    llama_model, llama_tokenizer = load_llama_with_adapter()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Function to generate responses
def generate_response(query):
    try:
        inputs = llama_tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = llama_model.generate(inputs["input_ids"], max_length=100, do_sample=True)
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Llama Bot", layout="wide")
st.title("💬 Llama 3.2 Conversational Bot")

user_query = st.text_input("Your Question", placeholder="💡 Ask me anything!")
if st.button("Generate"):
    if user_query:
        response = generate_response(user_query)
        st.markdown(f"**Response:** {response}")
