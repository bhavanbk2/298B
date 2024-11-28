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


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import os

class LlamaHandler:
    def __init__(self, model_path: str = "path/to/your/llama/model"):
        """
        Initialize the Llama model handler.
        
        Args:
            model_path: Path to the saved Llama model directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Move model to appropriate device
        self.model.to(self.device)
        
        # Set default parameters
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.95
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format chat history into a single string.
        
        Args:
            chat_history: List of dictionaries containing 'user' and 'bot' messages
        
        Returns:
            Formatted chat history string
        """
        formatted_history = ""
        for chat in chat_history:
            formatted_history += f"User: {chat['user']}\nAssistant: {chat['bot']}\n"
        return formatted_history
    
    def _format_prompt(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Format the input prompt with persona and chat history.
        
        Args:
            user_input: Current user message
            persona: Selected persona (e.g., "Friendly Assistant", "Technical Expert")
            chat_history: Previous conversation history
        
        Returns:
            Formatted prompt string
        """
        # Add persona-specific instructions
        persona_instructions = {
            "Friendly Assistant": "You are a friendly and helpful assistant. Respond in a casual, warm manner.",
            "Technical Expert": "You are a technical expert. Provide detailed, technical responses with precise information.",
            "Creative Writer": "You are a creative writer. Write engaging, imaginative responses with literary flair."
        }
        
        system_prompt = persona_instructions.get(persona, persona_instructions["Friendly Assistant"])
        chat_history_text = self._format_chat_history(chat_history)
        
        # Combine all elements into final prompt
        full_prompt = f"""{system_prompt}

Previous conversation:
{chat_history_text}

User: {user_input}
Assistant:"""
        
        return full_prompt
    
    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Generate a response using the Llama model.
        
        Args:
            user_input: User's message
            persona: Selected persona
            chat_history: Previous conversation history
        
        Returns:
            Generated response string
        """
        try:
            # Format the prompt
            prompt = self._format_prompt(user_input, persona, chat_history)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=self.max_length).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
