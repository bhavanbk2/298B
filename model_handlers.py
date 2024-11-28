import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import streamlit as st

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

class LlamaHandler(ModelHandler):
    def __init__(self):
        self.authenticate()
        self.initialize_model()
    
    def authenticate(self):
        try:
            hf_token = os.getenv("HUGGING_FACE_TOKEN")
            if not hf_token:
                st.error("Hugging Face token not found!")
                return False
            login(token=hf_token)
            return True
        except Exception as e:
            st.error(f"Hugging Face authentication failed: {str(e)}")
            return False
    
    def initialize_model(self):
        try:
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            
            if 'llama_pipeline' not in st.session_state:
                st.session_state.llama_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    tokenizer=model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        except Exception as e:
            st.error(f"Error initializing Llama model: {str(e)}")
    
    def generate_response(self, query, persona, chat_history):
        try:
            if 'llama_pipeline' not in st.session_state:
                return "Model not initialized. Please try again."
            
            context = " ".join([
                f"User: {item['user']} Assistant: {item['bot']}" 
                for item in chat_history[-3:]
            ])
            
            prompt = f"""<s>[INST] You are a {persona}.
            Previous conversation: {context}
            Current query: {query} [/INST]"""
            
            response = st.session_state.llama_pipeline(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            generated_text = response[0]['generated_text']
            return generated_text.split('[/INST]')[-1].strip()
            
        except Exception as e:
            st.error(f"Error with Llama: {str(e)}")
            return "Sorry, I encountered an error. Please try again."
