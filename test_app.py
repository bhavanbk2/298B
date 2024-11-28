import streamlit as st
import os
from dotenv import load_dotenv
from model_handlers import GPTHandler, LlamaHandler
import base64
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = None

def load_image(image_path):
    """Load and encode image to base64"""
    path = Path(image_path)
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None

def typing_animation():
    """Simulate typing animation"""
    with st.spinner('Bot is typing...'):
        time.sleep(1)

def apply_custom_css(theme):
    """Apply custom CSS based on theme"""
    primary_color = "#121212" if theme == "Dark" else "#f5f5f7"
    text_color = "white" if theme == "Dark" else "black"
    question_mark_color = "#6C757D"

    st.markdown(f"""
    <style>
    body {{
        background-color: {primary_color};
        color: {text_color};
        font-family: 'Arial', sans-serif;
    }}
    .stApp {{
        background-color: {primary_color};
    }}
    .chat-bubble {{
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
    }}
    .chat-bubble-user {{
        background-color: #DCF8C6;
        align-self: flex-end;
        color: black;
    }}
    .chat-bubble-bot {{
        background-color: #F1F0F0;
        align-self: flex-start;
        color: black;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }}
    .avatar {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }}
    .title-text {{
        font-size: 36px;
        font-weight: bold;
        color: {text_color};
    }}
    .subtitle-text {{
        font-size: 20px;
        color: {text_color};
        margin-top: -10px;
    }}
    .no-conversation {{
        font-size: 20px;
        color: #ffc107;
        text-align: center;
        margin-top: 20px;
        font-style: italic;
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Chat Assistant", layout="wide")
    
    # Sidebar
    with st.sidebar:
        if st.button("📝 New Chat"):
            st.session_state.chat_history.clear()
            st.session_state.model_handler = None
        
        st.markdown("### 🧠 Choose Assistant Personality")
        persona = st.selectbox("Select Persona", 
                             ["Friendly Assistant", "Technical Expert", "Creative Writer"])
        
        st.markdown("### 🤖 Choose AI Model")
        model_choice = st.selectbox("Select Model", 
                                  ["GPT-3.5", "Llama-2"],
                                  help="Choose your preferred AI model")
        
        st.markdown("### 🌗 Theme")
        theme = st.radio("Choose Theme", ["Dark", "Light"], index=0)
    
    # Apply theme
    apply_custom_css(theme)
    
    # Main content
    st.markdown("<h1 class='title-text'>💬 AI Chat Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Ask me anything!</p>", unsafe_allow_html=True)
    
    # Initialize model handler if not exists or model changed
    if model_choice == "GPT-3.5":
        if not isinstance(st.session_state.model_handler, GPTHandler):
            st.session_state.model_handler = GPTHandler()
    else:  # Llama-2
        if not isinstance(st.session_state.model_handler, LlamaHandler):
            st.session_state.model_handler = LlamaHandler()
    
    # Chat interface
    user_input = st.text_input("Your message:", key="user_input", 
                              placeholder="Type your message here...")
    
    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):
                response = st.session_state.model_handler.generate_response(
                    user_input, persona, st.session_state.chat_history
                )
                st.session_state.chat_history.append({
                    'user': user_input,
                    'bot': response
                })
    
    # Display chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['bot']}")
            st.markdown("---")
    else:
        st.markdown("<p class='no-conversation'>Start a conversation!</p>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()
