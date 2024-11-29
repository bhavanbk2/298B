import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login, list_repo_files, HfApi
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
    def verify_model_access(self, base_model_path: str) -> bool:
        """Verify access to the base model."""
        try:
            # Check for HF token
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                st.error("No Hugging Face token found in environment variables!")
                st.info("Please set your Hugging Face token in the environment variables.")
                return False
            
            # Try to login
            st.info("Authenticating with Hugging Face...")
            login(token=hf_token)
            
            # List files in the repository
            st.info(f"Verifying access to {base_model_path}...")
            files = list_repo_files(base_model_path)
            
            # Check if essential files are present
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            found_files = [f for f in files if f.split('/')[-1] in required_files]
            
            if not all(req in [f.split('/')[-1] for f in found_files] for req in required_files):
                st.warning("Some required model files are missing!")
                st.info("Files found: " + ", ".join(found_files))
                return False
            
            st.success("Successfully verified model access!")
            return True
            
        except Exception as e:
            st.error("Error during model verification:")
            st.error(str(e))
            st.info("""
            Common issues:
            1. Missing or invalid Hugging Face token
            2. No access to the Llama model (requires acceptance of Meta's license)
            3. Network connectivity issues
            
            To fix:
            1. Get your token from: https://huggingface.co/settings/tokens
            2. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
            3. Accept the model's license agreement
            """)
            return False

    def __init__(self):
        """Initialize the Llama model handler."""
        try:
            st.info("Initializing Llama model... This may take a few moments.")
            
            # Model paths
            adapter_model_path = "shashikumar1998/Llama-3.2-3B-Instruct"
            base_model_path = "meta-llama/Llama-2-7b-chat-hf"
            
            # Verify base model access first
            if not self.verify_model_access(base_model_path):
                raise ValueError("Could not access the base model. Please check your Hugging Face token and model access.")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {self.device}")
            
            # Load tokenizer from base model first
            st.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            st.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                adapter_model_path,
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
            st.error("Detailed error information for debugging:")
            st.error(f"Type: {type(e)}")
            st.error(f"Args: {e.args}")
            raise
    
    def _format_prompt(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Format the input prompt."""
        system_prompt = f"You are a {persona}. Be helpful and concise."
        
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
