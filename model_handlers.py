import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
import streamlit as st

class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
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
    
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        try:
            context = " ".join([
                f"User: {item['user']} Bot: {item['bot']}" 
                for item in chat_history[-3:]  # Use last 3 interactions
            ])
            
            messages = [
                {"role": "system", "content": f"You are {persona}. Remember you are not to respond outside the core competency of the {persona}, but still be able to respond any basic queries. Make sure to impart the {persona} and respond like how the person does respond to each and every differnent question in a unique way. Respond accordingly."},
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
            st.info("Initializing Llama handler...")
            # configuration for Llama API
            self.api_url = "https://api-inference.huggingface.co/models/llama-model"
            self.headers = {
                "Authorization": llama_key,
                "Content-Type": "application/json"
            } self. initialized = True
        except Exception:
            self.initialized = False
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.gpt_handler.generate_response(query, persona, chat_history)

        try:
            payload = {
                "inputs": f"<s>[INST] Act as {persona}. {query} [/INST]",
                "parameters": {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
            }
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json()[0]['generated_text']
            else:
                return self.gpt_handler.generate_response(query, persona, chat_history)
        except Exception:
            return self.gpt_handler.generate_response(query, persona, chat_history)


class GemmaHandler(ModelHandler):
    def __init__(self):
        try:
            st.info("Initializing Gemma handler...")
            # configuration for Gemma API
            self.api_url = "https://api-inference.huggingface.co/models/gemma-model"
            self.headers = {
                "Authorization": gemma_key,
                "Content-Type": "application/json"
            } self.initialized = True
        except Exception:
            self.initialized = False

    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.gpt_handler.generate_response(query, persona, chat_history)

        try:
            payload = {
                "input": f"Persona: {persona}\nQuery: {query}",
                "parameters": {"max_tokens": 150}
            }
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get('text', '')
            else:
                return self.gpt_handler.generate_response(query, persona, chat_history)
        except Exception:
            return self.gpt_handler.generate_response(query, persona, chat_history)

class PalmHandler(ModelHandler):
    def __init__(self):
        try:
            st.info("Initializing Palm handler...")
            # configuration for Palm API
            self.api_url = "https://api-inference.huggingface.co/models/palm-model"
            self.headers = {
                "Authorization": palm_key,
                "Content-Type": "application/json"
            } self.initialized = True
        except Exception:
            self.initialized = False
    def generate_response(self, query: str, persona: str, chat_history: list) -> str:
        if not self.initialized:
            return self.gpt_handler.generate_response(query, persona, chat_history)

        try:
            payload = {
                "prompt": f"Persona: {persona}\nChat history: {chat_history}\nUser Query: {query}",
                "options": {"max_output_tokens": 256}
            }
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return self.gpt_handler.generate_response(query, persona, chat_history)
        except Exception:
            return self.gpt_handler.generate_response(query, persona, chat_history)
