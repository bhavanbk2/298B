import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
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
        """Initialize the Llama model handler."""
        try:
            st.info("Initializing Llama model... This may take a few moments.")
            
            model_path = "sainathv02/llama3_1_insurance_qlora"
            base_model_path = "meta-llama/Llama-2-7b-chat-hf"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {self.device}")
            
            # Create base Llama configuration
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                hidden_act="silu",
                max_position_embeddings=4096,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                pretraining_tp=1,
                tie_word_embeddings=False,
                rope_scaling={"type": "dynamic", "factor": 2.0}
            )
            
            # Load tokenizer
            st.info("Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
            except Exception as e:
                st.warning(f"Failed to load custom tokenizer, falling back to base model tokenizer: {str(e)}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with configuration
            st.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Generation parameters
            self.max_length = 512
            self.temperature = 0.7
            self.top_p = 0.95
            
            st.success("Llama model loaded successfully!")
            
        except Exception as e:
            st.error(f"Error initializing Llama model: {str(e)}")
            raise
    
    def _format_prompt(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Format the input prompt."""
        # Simplified prompt template for insurance-specific model
        system_prompt = f"You are a {persona} specializing in insurance. Be helpful and concise."
        
        # Include only last 2 interactions for context
        recent_history = chat_history[-2:] if chat_history else []
        history_text = "\n".join([
            f"User: {chat['user']}\nAssistant: {chat['bot']}"
            for chat in recent_history
        ])
        
        # Format for Llama
        return f"""<s>[INST] {system_prompt}

Previous conversation:
{history_text}

User: {user_input} [/INST]"""
    
    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the Llama model."""
        try:
            # Format prompt
            prompt = self._format_prompt(user_input, persona, chat_history)
            
            # Tokenize with proper padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            # Generate with error handling
            with torch.no_grad():
                try:
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
                    
                    # Extract only the assistant's response
                    if "[/INST]" in response:
                        response = response.split("[/INST]")[-1].strip()
                    
                    # Truncate if too long
                    if len(response) > 1000:
                        response = response[:1000] + "..."
                    
                    return response
                    
                except torch.cuda.OutOfMemoryError:
                    st.warning("GPU memory exceeded. Falling back to CPU...")
                    self.model.to("cpu")
                    self.device = "cpu"
                    return self.generate_response(user_input, persona, chat_history)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."
