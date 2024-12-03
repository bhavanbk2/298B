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

def apply_custom_css(theme):
    """Apply custom CSS based on theme."""
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"

    st.markdown(f"""
    <style>
    body {{
        background-color: {primary_color};
        color: {text_color};
    }}
    .stApp {{
        background-color: {primary_color};
    }}
    .chat-item {{
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }}
    .chat-avatar {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }}
    .chat-text {{
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        color: black;
    }}
    .chat-text-user {{
        background-color: #DCF8C6;
    }}
    </style>
    """, unsafe_allow_html=True)

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

def load_image(image_path):
    """Load and return an image."""
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

def main():
    st.set_page_config(page_title="AI Chat Assistant with Images", layout="wide")

    # Sidebar
    with st.sidebar:
        if st.button("📝 New Chat"):
            st.session_state.chat_history.clear()
            st.session_state.model_handler = None
        
        st.markdown("### 🧠 Choose Assistant Personality")
        persona = st.selectbox(
            "Select Persona",
            ["Sanjay Gupta", "Robert Kiyosaki"],
            help="Choose the assistant's persona"
        )
        st.session_state.persona = persona

        st.markdown("### 🤖 Choose AI Model")
        model_choice = st.selectbox(
            "Select Model",
            ["GPT-3.5", "Llama", "Gemma", "Palm"],
            help="Choose your preferred AI model"
        )

        st.markdown("### 🌗 Theme")
        theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)

    # Apply theme
    apply_custom_css(theme)

    # Main content
    st.title("💬 Persona-Based Conversational Bot")
    st.markdown(f"Ask your question to **{st.session_state.persona}**.")
    
    # Initialize model handler if not exists or model changed
    if model_choice == "GPT-3.5":
        if not isinstance(st.session_state.model_handler, GPTHandler):
            st.session_state.model_handler = GPTHandler()
    elif model_choice == "Llama":
        if not isinstance(st.session_state.model_handler, LlamaHandler):
            st.session_state.model_handler = LlamaHandler()
    elif model_choice == "Gemma":
        if not isinstance(st.session_state.model_handler, GemmaHandler):
            st.session_state.model_handler = GemmaHandler()
    elif model_choice == "Palm":
        if not isinstance(st.session_state.model_handler, PalmHandler):
            st.session_state.model_handler = PalmHandler()

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
            # User Message with Image
            st.markdown("<div class='chat-item'>", unsafe_allow_html=True)
            st.image("images/user_image.png", width=40, caption="User")
            st.markdown(f"<div class='chat-text chat-text-user'>{chat['user']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Bot Message with Image
            st.markdown("<div class='chat-item'>", unsafe_allow_html=True)
            st.image("images/bot_image.png", width=40, caption="Bot")
            st.markdown(f"<div class='chat-text'>{chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Export Chat History Button
        export_chat_history()
    else:
        st.markdown("<p style='color: #6c757d;'>Start a conversation!</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
