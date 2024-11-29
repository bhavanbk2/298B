import requests
import streamlit as st
from typing import List, Dict
import json

class LlamaAPIHandler:
    def __init__(self):
        """Initialize the API handler."""
        try:
            # Add your token here
            self.api_token = "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"  # Replace with your actual token
            
            # API configuration
            self.api_url = "https://api-inference.huggingface.co/models/shashikumar1998/Llama-3.2-3B-Instruct"
            self.headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            st.success("API handler initialized successfully!")
            
        except Exception as e:
            st.error(f"Error initializing API handler: {str(e)}")
            raise
    
    def query_api(self, prompt: str) -> str:
        """Send a query to the Hugging Face API."""
        try:
            # Basic payload structure for text generation
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Debug payload
            st.write("Sending payload:")
            st.write(payload)
            
            # Send request to API
            with st.spinner("Generating response..."):
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
            
            # Debug response
            st.write(f"Response Status: {response.status_code}")
            st.write("Response Content:")
            st.write(response.text)
            
            # Check response
            if response.status_code == 200:
                response_data = response.json()
                st.write("Parsed Response:")
                st.write(response_data)
                
                if isinstance(response_data, list) and len(response_data) > 0:
                    return response_data[0].get('generated_text', '')
                return str(response_data)
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                st.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server might be busy.")
            return "I apologize, but the request timed out. Please try again."
            
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            return "I apologize, but I encountered an error communicating with the API."

    def generate_response(self, user_input: str, persona: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response using the API."""
        try:
            # Format prompt - simplified version
            prompt = f"<s>[INST] Act as {persona}. {user_input} [/INST]"
            
            # Get response from API
            response = self.query_api(prompt)
            
            # Clean up response
            response = response.strip()
            
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        persona = st.selectbox(
            "Choose Assistant Persona",
            ["Helpful Assistant", "Technical Expert", "Creative Writer"]
        )
        
        # Add debug toggle
        st.session_state.debug = st.checkbox("Show Debug Info", value=False)
    
    # Initialize model handler
    if st.session_state.model_handler is None:
        try:
            st.session_state.model_handler = LlamaAPIHandler()
        except Exception as e:
            st.error("Failed to initialize API handler.")
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
                st.session_state.chat_history[:-1]
            )
            
            # Update last message with bot response
            st.session_state.chat_history[-1]['bot'] = response
    
    # Display chat history
    st.markdown("### Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"👤 **You:** {chat['user']}")
        if chat['bot']:
            st.markdown(f"🤖 **Assistant:** {chat['bot']}")
        st.markdown("---")

if __name__ == "__main__":
    main()
