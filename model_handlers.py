import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import streamlit as st
from typing import List, Dict

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
        """Initialize the TinyLlama model handler."""
        try:
            st.info("Initializing Llama model... This may take a few moments.")
            
            # Use TinyLlama model
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {self.device}")
            
            # Create the pipeline directly
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                model_kwargs={"low_cpu_mem_usage": True}
            )
            
            st.success("Model loaded successfully!")
            
        except Exception as e:
            st.error(f"Error initializing Llama model: {str(e)}")
            raise
    
    def _format_prompt(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Format the input prompt."""
        # Simplified prompt template
        system_prompt = f"You are a {persona}. Be helpful and concise."
        
        # Include only last 2 interactions for context
        recent_history = chat_history[-2:] if chat_history else []
        history_text = "\n".join([
            f"User: {chat['user']}\nAssistant: {chat['bot']}"
            for chat in recent_history
        ])
        
        # TinyLlama specific prompt format
        return f"<|system|>{system_prompt}</s><|user|>{history_text}\n{user_input}</s><|assistant|>"
    
    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the model."""
        try:
            # Format prompt
            prompt = self._format_prompt(user_input, persona, chat_history)
            
            # Generate response using pipeline
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.pad_token_id if hasattr(self.pipeline.tokenizer, 'pad_token_id') else None,
            )[0]['generated_text']
            
            # Extract assistant's response
            response_parts = response.split("<|assistant|>")
            if len(response_parts) > 1:
                response = response_parts[-1].split("</s>")[0].strip()
            else:
                response = response_parts[0].strip()
            
            # Truncate if too long
            if len(response) > 1000:
                response = response[:1000] + "..."
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            st.warning("GPU memory exceeded. Falling back to CPU...")
            self.pipeline.device = torch.device('cpu')
            return self.generate_response(user_input, persona, chat_history)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."
