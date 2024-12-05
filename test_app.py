import streamlit as st
import os
from dotenv import load_dotenv
from model_handlers import GPTHandler, LlamaHandler, GemmaHandler, PalmHandler
import time
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = None
if 'persona' not in st.session_state:
    st.session_state.persona = "Friendly Assistant"

def typing_animation():
    """Simulate a typing animation."""
    with st.spinner("Assistant is typing..."):
        time.sleep(1)

def export_chat_history():
    """Export chat history as a .txt file."""
    history = "\n\n".join(
        [f"User: {chat['user']}\nAssistant: {chat['bot']}" for chat in st.session_state.chat_history]
    )
    st.download_button(
        label="📥 Download Chat History",
        data=history,
        file_name="chat_history.txt",
        mime="text/plain"
    )

def main():
    st.set_page_config(page_title="AI Chat Assistant", layout="wide")

    # Sidebar
    with st.sidebar:
        if st.button("📝 New Chat"):
            st.session_state.chat_history.clear()
            st.session_state.model_handler = None
            st.success("Chat history cleared successfully!")
        
        st.markdown("### 🧠 Choose Assistant Personality")
        new_persona = st.selectbox(
            "Select Persona",
            ["Sanjay Gupta", "Robert Kiyosaki"],
            help="Choose the assistant's persona"
        )
        if new_persona != st.session_state.persona:
            st.session_state.chat_history.clear()
            st.session_state.persona = new_persona
            st.success(f"Persona updated to {new_persona}!")

        st.markdown("### 🤖 Choose AI Model")
        new_model_choice = st.selectbox(
            "Select Model",
            ["GPT", "Llama", "Gemma", "Palm"],
            help="Choose your preferred AI model"
        )
        if st.session_state.model_handler is None or type(st.session_state.model_handler).__name__ != f"{new_model_choice}Handler":
            st.session_state.chat_history.clear()
            if new_model_choice == "GPT":
                st.session_state.model_handler = GPTHandler()
            elif new_model_choice == "Llama":
                st.session_state.model_handler = LlamaHandler()
            elif new_model_choice == "Gemma":
                st.session_state.model_handler = GemmaHandler()
            elif new_model_choice == "Palm":
                st.session_state.model_handler = PalmHandler()
            st.success(f"{new_model_choice} model loaded successfully!")

    # Main content
    st.title("💬 Persona-Based Conversational Bot")
    st.markdown(f"Ask your question to **{st.session_state.persona}**.")
    
    # Chat interface
    user_input = st.text_input("Your message:", key="user_input", placeholder="Type your question here...")
    
    if st.button("Send"):
        if user_input:
            typing_animation()
            try:
                # Generate response
                response = st.session_state.model_handler.generate_response(
                    user_input, st.session_state.persona, st.session_state.chat_history
                )
                st.session_state.chat_history.append({
                    'user': user_input,
                    'bot': response
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            # User Message
            st.markdown(f"**You:** {chat['user']}")
            # Bot Message
            st.markdown(f"**{st.session_state.persona}:** {chat['bot']}")
            st.markdown("---")
            
        # Export Chat History Button
        export_chat_history()
    else:
        st.markdown("<p style='color: #6c757d;'>Start a conversation!</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
