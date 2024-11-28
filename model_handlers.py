import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import streamlit as st
import os
from abc import ABC, abstractmethod
import streamlit as st
import torch

# Unsloth imports
try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError("Please install unsloth: pip install --upgrade --no-cache-dir 'unsloth[cuda] @ git+https://github.com/unslothai/unsloth.git'")

# Transformers import
from transformers import TextStreamer

# RAG-related imports
try:
    import cohere
    from pinecone import Pinecone
except ImportError:
    raise ImportError("Please install required packages: pip install cohere pinecone-client")

class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query, persona, chat_history):
        pass

class GPTHandler(ModelHandler):
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
        
    
    def generate_response(self, query, persona, chat_history):
        try:
            context = " ".join([
                f"User: {item['user']} Bot: {item['bot']}" 
                for item in chat_history[-3:]  # Last 3 interactions
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


class OptimizedLlamaHandler(ModelHandler):
    class OptimizedLlamaHandler(ModelHandler):
    def __init__(self):
        self.authenticate()
        self.initialize_model()
    
    def authenticate(self):
        try:
            self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
            if not self.hf_token:
                st.error("Hugging Face token not found!")
                return False
            login(token=self.hf_token)
            return True
        except Exception as e:
            st.error(f"Hugging Face authentication failed: {str(e)}")
            return False
            
    def initialize_model(self):
        try:
            if 'llama_model' not in st.session_state:
                max_seq_length = 3072  # You can adjust this
                dtype = None  # Auto detection
                
                model_and_tokenizer = FastLanguageModel.from_pretrained(
                    model_name="shashikumar1998/Llama-3.2-3B-Instruct",
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    token=self.hf_token
                )

                if isinstance(model_and_tokenizer, tuple):
                    st.session_state.llama_model = model_and_tokenizer[0]
                    st.session_state.llama_tokenizer = model_and_tokenizer[1]
                else:
                    st.session_state.llama_model = model_and_tokenizer
                    
        except Exception as e:
            st.error(f"Error initializing Llama model: {str(e)}")
    
    def generate_response(self, query, persona, chat_history):
        try:
            if not hasattr(st.session_state, 'llama_model'):
                return "Model not initialized. Please try again."
            
            # Get context from recent chat history
            context = " ".join([
                f"User: {item['user']} Assistant: {item['bot']}" 
                for item in chat_history[-3:]
            ])
            
            # Prepare the prompt
            prompt = f"""<s>[INST] You are a {persona}.
            Previous conversation: {context}
            Current query: {query} [/INST]"""
            
            # Prepare inputs
            inputs = st.session_state.llama_tokenizer(
                [prompt],
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Generate response
            with st.spinner("Generating response..."):
                # For inference
                FastLanguageModel.for_inference(st.session_state.llama_model)
                
                outputs = st.session_state.llama_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    use_cache=True,
                    temperature=0.7,
                    min_p=0.1
                )
                
                response = st.session_state.llama_tokenizer.batch_decode(outputs)[0]
                return response.split('[/INST]')[-1].strip()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Please try again."
