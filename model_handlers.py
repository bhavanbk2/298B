import os
import requests
from abc import ABC, abstractmethod
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI


class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        pass


class BaseHandler(ModelHandler):
    def __init__(self):
        """Initialize the GPT fallback handler."""
        self.gpt_handler = GPTHandler()

    def fallback_to_gpt(self, query, persona, chat_history):
        """Fallback to GPT handler."""
        return self.gpt_handler.generate_response(query, persona, chat_history)


class GPTHandler:
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
    
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        try:
            context = " ".join([
                f"User: {item['user']} Bot: {item['bot']}" 
                for item in chat_history[-3:]  # Use last 3 interactions
            ])
            
            messages = [
                {"role": "system", "content": f"You are {persona}. Respond uniquely and provide accurate assistance."},
                {"role": "user", "content": f"{context} {query}"}
            ]
            
            response = self.client(messages)
            return response.content
        except Exception as e:
            st.error(f"Error with GPT: {str(e)}")
            return "Sorry, I encountered an error. Please try again."


class LlamaHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        try:
            st.info("Initializing Llama handler...")
            self.api_url = "https://api-inference.huggingface.co/models/llama-model"
            self.headers = {
                "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",
                "Content-Type": "application/json"
            }
            self.initialized = True
        except Exception as e:
            self.initialized = False
            st.error(f"Failed to initialize LlamaHandler: {e}")

    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.fallback_to_gpt(query, persona, chat_history)

        try:
            payload = {
                "inputs": f"<s>[INST] Act as {persona}. {query} [/INST]",
                "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()[0].get('generated_text', 'Error: No response from Llama.')
            else:
                return self.fallback_to_gpt(query, persona, chat_history)
        except Exception as e:
            st.error(f"Error with Llama API: {e}")
            return self.fallback_to_gpt(query, persona, chat_history)


class GemmaHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        try:
            st.info("Initializing Gemma handler...")
            self.api_url = "https://api-inference.huggingface.co/models/gemma-model"
            self.headers = {
                "Authorization": f"Bearer {os.getenv('GEMMA_API_KEY')}",
                "Content-Type": "application/json"
            }
            self.initialized = True
        except Exception as e:
            self.initialized = False
            st.error(f"Failed to initialize GemmaHandler: {e}")

    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.fallback_to_gpt(query, persona, chat_history)

        try:
            payload = {
                "input": f"Persona: {persona}\nQuery: {query}",
                "parameters": {"max_tokens": 150}
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get('text', 'Error: No response from Gemma.')
            else:
                return self.fallback_to_gpt(query, persona, chat_history)
        except Exception as e:
            st.error(f"Error with Gemma API: {e}")
            return self.fallback_to_gpt(query, persona, chat_history)


class PalmHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        try:
            st.info("Initializing Palm handler...")
            self.api_url = "https://api-inference.huggingface.co/models/palm-model"
            self.headers = {
                "Authorization": f"Bearer {os.getenv('PALM_API_KEY')}",
                "Content-Type": "application/json"
            }
            self.initialized = True
        except Exception as e:
            self.initialized = False
            st.error(f"Failed to initialize PalmHandler: {e}")

    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.fallback_to_gpt(query, persona, chat_history)

        try:
            payload = {
                "prompt": f"Persona: {persona}\nChat history: {chat_history}\nUser Query: {query}",
                "options": {"max_output_tokens": 256}
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get('response', 'Error: No response from Palm.')
            else:
                return self.fallback_to_gpt(query, persona, chat_history)
        except Exception as e:
            st.error(f"Error with Palm API: {e}")
            return self.fallback_to_gpt(query, persona, chat_history)
