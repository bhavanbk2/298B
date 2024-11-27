import streamlit as st
import os
from dotenv import load_dotenv
import cohere
import pinecone
from unsloth import FastLanguageModel
from transformers import TextStreamer
import json
import time
import base64

# Load environment variables
load_dotenv()

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(api_key=cohere_api_key)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
index = pinecone.Index("cohere-pinecone-tree")

# Load Hugging Face token
hf_token = os.getenv("Hugging_face_token")

# Load Llama 3.2 model and tokenizer
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True  # Optimize memory usage

@st.cache_resource
def load_llama_model():
    model_and_tokenizer = FastLanguageModel.from_pretrained(
        model_name="shashikumar1998/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token,
    )
    if isinstance(model_and_tokenizer, tuple):
        return model_and_tokenizer[0], model_and_tokenizer[1]
    return model_and_tokenizer, None

model, tokenizer = load_llama_model()

# Function to generate RAG response
def generate_rag_response(query):
    try:
        # Step 1: Generate query embedding using Cohere
        response = cohere_client.embed(texts=[query], model="embed-english-light-v2.0")
        query_embedding = response.embeddings[0]

        # Step 2: Retrieve relevant documents from Pinecone
        top_k = 5
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieved_context = "\n".join([result["metadata"]["text"] for result in results["matches"]])

        # Step 3: Prepare input for Llama model
        messages = [
            {"role": "system", "content": f"Context: {retrieved_context}"},
            {"role": "user", "content": query},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # Step 4: Generate response
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            use_cache=True,
            temperature=0.5,
            min_p=0.1,
            max_new_tokens=64,
        )
        return " ".join(text_streamer.history)  # Combine streamed tokens
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history = []

    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox(
        "Select Persona", ["Robert Kiyosaki", "Sanjay Gupta"]
    )

    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox("Select Model", ["Llama 3.2 (RAG)", "Other Models"])

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

# Chat input and output
st.title("💬 Persona-Based Conversational Bot")
user_query = st.text_input("Your Question", placeholder="💡 Ask a question")

if st.button("Submit"):
    if user_query:
        if model_choice == "Llama 3.2 (RAG)":
            response = generate_rag_response(user_query)
        else:
            response = "Model not implemented."

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append({"user": user_query, "bot": response})

# Display chat history
if "chat_history" in st.session_state:
    for interaction in st.session_state.chat_history:
        st.markdown(f"**User:** {interaction['user']}")
        st.markdown(f"**Bot:** {interaction['bot']}")
