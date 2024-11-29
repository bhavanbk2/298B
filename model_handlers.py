import os
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama import LlamaConfig
from transformers.utils import logging
from peft import PeftModel, PeftConfig
import torch
import streamlit as st
from typing import List, Dict
from langchain_openai.chat_models import ChatOpenAI
import os
from abc import ABC, abstractmethod
from unsloth import FastLanguageModel
import cohere
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoConfig, TextStreamer
from typing import List, Dict

# Disable unnecessary warnings
logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Initialize constants
MAX_SEQ_LENGTH = 2048  # Sequence length for the model
LOAD_IN_4BIT = True    # Enable 4-bit quantization
DTYPE = None           # Auto-detect or specify float16, bfloat16, etc.
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize Cohere and Pinecone API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        pass

class GPTHandler(ModelHandler):
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            self.client = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo"
            )
        except Exception as e:
            st.error(f"Error initializing GPT handler: {str(e)}")
            raise
    
    def generate_response(self, query: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        try:
            context = " ".join([
                f"User: {item['user']} Bot: {item['bot']}" 
                for item in chat_history[-3:]
            ])
            
            messages = [
                {"role": "system", "content": f"You are a {persona}. Respond accordingly."},
                {"role": "user", "content": f"{context} {query}"}
            ]
            
            response = self.client(messages)
            return response.content
            
        except Exception as e:
            st.error(f"Error with GPT: {str(e)}")
            return "Sorry, I encountered an error. Please try again."

class LlamaHandler(ModelHandler):
    def __init__(self):
        try:
            # Load the model and tokenizer
            self.model_and_tokenizer = FastLanguageModel.from_pretrained(
                model_name="shashikumar1998/Llama-3.2-3B-Instruct",
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=DTYPE,
                load_in_4bit=LOAD_IN_4BIT,  # Disable 4-bit quantization for CPU
                token=HF_TOKEN
            )

            # Unpack model and tokenizer if returned as a tuple
            if isinstance(self.model_and_tokenizer, tuple):
                self.model = self.model_and_tokenizer[0]
                self.tokenizer = self.model_and_tokenizer[1]
            else:
                self.model = self.model_and_tokenizer
                self.tokenizer = None  # Adjust based on library response

            # Move model to the detected device (CPU/GPU)
            self.model.to(DEVICE)

            # Initialize Cohere and Pinecone
            self.cohere_client = cohere.Client(COHERE_API_KEY)
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index("cohere-pinecone-tree")

            print(f"Model initialized successfully on device: {DEVICE}")

        except Exception as e:
            print(f"Error initializing FastLanguageModelHandler: {e}")
            raise

    def generate_rag_response(self, query: str) -> str:
        try:
            # Step 1: Generate query embedding using Cohere
            response = self.cohere_client.embed(texts=[query], model="embed-english-light-v2.0")
            query_embedding = response.embeddings[0]

            # Step 2: Retrieve relevant documents from Pinecone
            top_k = 5  # Number of documents to retrieve
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            retrieved_context = "\n".join([result["metadata"]["text"] for result in results["matches"]])

            # Step 3: Prepare input for the model
            messages = [
                {"role": "system", "content": f"Here is some context to help answer the question: {retrieved_context}"},
                {"role": "user", "content": query},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(DEVICE)

            # Step 4: Generate response using the model
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=64,
                use_cache=True,
                temperature=0.5,
                min_p=0.1,
            )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return response_text

        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return "I encountered an error. Please try again."
