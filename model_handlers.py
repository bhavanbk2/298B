import os
from abc import ABC, abstractmethod
# Fix imports from transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedTokenizer
)
import torch
import streamlit as st
from typing import List, Dict
import os
from abc import ABC, abstractmethod
# Fix imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from transformers import BitsAndBytesConfig  # Added this import
from peft import PeftModel, PeftConfig
import torch
import streamlit as st
from typing import List, Dict

# Disable unnecessary warnings
logging.set_verbosity_error()


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
        """Initialize the PEFT-adapted Llama model handler."""
        try:
            st.info("Initializing model...")
            
            adapter_path = "shashikumar1998/Llama-3.2-3B-Instruct"
            
            # Load PEFT config first to get base model name
            st.info("Loading PEFT configuration...")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Configure quantization
            st.info("Setting up quantization configuration...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )
            
            # Load base model with quantization
            st.info(f"Loading base model from {base_model_name}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # Load adapter weights
            st.info("Loading adapter weights...")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                device_map="auto"
            )
            
            # Load tokenizer
            st.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set generation parameters
            self.max_length = 512
            self.temperature = 0.7
            self.top_p = 0.95
            
            st.success("Model initialized successfully!")
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.error(f"Error type: {type(e)}")
            st.error(f"Base model name: {base_model_name}")
            raise

    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the model."""
        try:
            # Format prompt
            system_prompt = f"You are a {persona}. Be helpful and concise."
            
            # Include only last 2 interactions for context
            recent_history = chat_history[-2:] if chat_history else []
            history_text = "\n".join([
                f"User: {chat['user']}\nAssistant: {chat['bot']}"
                for chat in recent_history
            ])
            
            # Create full prompt
            prompt = f"""<s>[INST] {system_prompt}

Previous conversation:
{history_text}

User: {user_input} [/INST]"""

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                response = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            
            # Clean up response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            
            # Truncate if too long
            if len(response) > 1000:
                response = response[:1000] + "..."
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."
