import os
import requests
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Load environment variables
load_dotenv()

class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        pass

class LlamaAPIHandler(ModelHandler):
    def __init__(self):
        """Initialize the API handler."""
        try:
            # Get API token from environment variables
            self.api_token = os.getenv("HUGGINGFACE_TOKEN")
            if not self.api_token:
                raise ValueError("Hugging Face API token not found in environment variables")
            
            # API configuration
            self.api_url = "https://api-inference.huggingface.co/models/shashikumar1998/Llama-3.2-3B-Instruct"
            self.headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            # Generation parameters
            self.parameters = {
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
            
            st.success("API handler initialized successfully!")
            
        except Exception as e:
            st.error(f"Error initializing API handler: {str(e)}")
            raise

    def query_api(self, prompt: str) -> str:
        """Send a query to the Hugging Face API."""
        try:
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": self.parameters
            }
            
            # Send request to API
            with st.spinner("Generating response..."):
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                return str(result)
            else:
                st.error(f"API Error: {response.status_code}")
                return f"Error: Unable to generate response (status code {response.status_code})."
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server might be busy.")
            return "I apologize, but the request timed out. Please try again."
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return "I apologize, but I encountered an error communicating with the API."
    
    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the API."""
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

            # Get response from API
            response = self.query_api(prompt)
            
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

def main():
    st.title("💬 Chat Assistant")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        persona = st.selectbox(
            "Choose Assistant Persona",
            ["Helpful Assistant", "Technical Expert", "Creative Writer"]
        )
    
    # Initialize model handler if not already done
    if st.session_state.model_handler is None:
        try:
            st.session_state.model_handler = LlamaAPIHandler()
        except Exception as e:
            st.error("Failed to initialize API handler. Please check your Hugging Face token.")
            st.error(str(e))
            return
    
    # Chat interface
    user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here...")
    
    if st.button("Send", type="primary"):
        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({
                'user': user_input,
                'bot': None
            })
            
            # Generate response
            response = st.session_state.model_handler.generate_response(
                user_input,
                persona,
                st.session_state.chat_history[:-1]  # Exclude current message
            )
            
            # Update last message with bot response
            st.session_state.chat_history[-1]['bot'] = response
    
    # Display chat history
    st.markdown("### Chat History")
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f"👤 **You:** {chat['user']}")
        
        # Bot response
        if chat['bot']:
            st.markdown(f"🤖 **Assistant:** {chat['bot']}")
        
        st.markdown("---")

if __name__ == "__main__":
    main()
