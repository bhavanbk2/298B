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

# Dummy handlers using GPT logic
class LlamaHandler(GPTHandler):
    pass

class GemmaHandler(GPTHandler):
    pass

class PalmHandler(GPTHandler):
    pass
