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

def apply_theme_based_on_browser():
    """Automatically apply theme based on the browser's theme."""
    theme_js = """
    <script>
    (function() {
        const isDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
        const themeColor = isDarkMode ? "Dark" : "Light";
        window.parent.document.body.setAttribute("data-theme", themeColor);
    })();
    </script>
    """
    st.markdown(theme_js, unsafe_allow_html=True)

    # Fetch current theme dynamically
    browser_theme = st.session_state.get('theme', 'Light')
    primary_color = "#121212" if browser_theme == "Dark" else "#f5f5f7"
    text_color = "white" if browser_theme == "Dark" else "black"

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

def confirm_clear_chat_history(key: str):
    """Clear chat history with confirmation."""
    return st.button("Are you sure you want to clear the chat history?", key=key)

def load_image(image_path):
    """Load and return an image."""
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

def main():
    st.set_page_config(page_title="AI Chat Assistant with Dynamic Theme", layout="wide")

    # Apply browser-based theme
    apply_theme_based_on_browser()

    # Sidebar
    with st.sidebar:
        if st.button("📝 New Chat", key="new_chat"):
            if confirm_clear_chat_history(key="new_chat_confirm"):
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
            if confirm_clear_chat_history(key="persona_change"):
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
            if confirm_clear_chat_history(key="model_change"):
                st.session_state.chat_history.clear()
                if new_model_choice == "GPT":
                    st.session_state.model_handler = GPTHandler()
                elif new_model_choice == "Llama":
                    st.session_state.model_handler = LlamaHandler()
                elif new_model_choice == "Gemma":
                    st.session_state.model_handler = GemmaHandler()
                elif new_model_choice == "Palm":
                    st.session_state.model_handler = PalmHandler()
                st.success(f"Model switched to {new_model_choice}!")

    # Main content
    st.title("💬 Persona-Based Conversational Bot")
    st.markdown(f"Ask your question to **{st.session_state.persona}**.")
    
    # Chat interface
    user_input = st.text_input("Your message:", key="user_input", placeholder="Type your question here...")
    
    if st.button("Send", key="send_message"):
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
