import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_openai.chat_models import ChatOpenAI
import embedding as emb
import json
import time
import base64

# Load environment variables
load_dotenv()

# Initialize Cohere client
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# Initialize LangChain's ChatOpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Get the embedding index
index = emb.get_index("cohere-pinecone-tree")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'persona' not in st.session_state:
    st.session_state.persona = "Friendly Assistant"

# Function to generate responses with dynamic persona-based system prompt
def generate_response(query):
    # Define persona-based system prompts
    persona_prompts = {
        "Sanjay Gupta": "You are a chatbot impersonating Dr. Sanjay Gupta, answering health and wellness questions.",
        "Motivational Coach": "You are a motivational coach chatbot, providing encouragement and advice on personal development.",
        "Friendly Assistant": "You are a friendly and helpful assistant chatbot, here to help with everyday queries.",
        "Robert Kiyosaki": "You are impersonating Robert Kiyosaki, offering advice on finance, investments, and wealth-building strategies."
    }
    
    # Get the system prompt based on the selected persona
    persona_prompt = persona_prompts.get(st.session_state.persona, "You are a chatbot.")
    
    # Create the context from chat history
    context = " ".join([f"User: {item['user']} Bot: {item['bot']}" for item in st.session_state.chat_history])
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": f"{context} {query}"}
    ]
    
    typing_animation()  # Simulate typing

    try:
        response = client(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        st.session_state.chat_history.append({'user': query, 'bot': response_text})
        return response_text

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Add a typing animation to simulate response generation
def typing_animation():
    with st.spinner('Bot is typing...'):
        time.sleep(1)  # Simulate typing delay

# Load images from the repository
def load_image(image_path):
    if os.path.exists(image_path):
        return base64.b64encode(open(image_path, "rb").read()).decode()
    return None

# Streamlit UI setup
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Sidebar for persona and theme selection
with st.sidebar:
    if st.button("📝 New Chat"):
        st.session_state.chat_history.clear()
    st.markdown("### 🧠 Choose Assistant Personality")
    st.session_state.persona = st.selectbox("Select Persona", ["Sanjay Gupta", "Motivational Coach", "Friendly Assistant", "Robert Kiyosaki"])
    st.markdown("### 🌗 Toggle Theme")
    theme = st.radio("Choose Theme", ["Dark", "Light"], index=1)  # Default to Light

# Load images from repository
user_avatar_path = "images/user_image.png"
bot_avatar_path = "images/bot_image.png"

user_avatar_base64 = load_image(user_avatar_path)
bot_avatar_base64 = load_image(bot_avatar_path)

# Ensure avatars are available
if user_avatar_base64 is None or bot_avatar_base64 is None:
    st.error("User and/or Bot avatars are missing. Please ensure the images are in the 'images' folder.")
    st.stop()

# CSS for enhanced UI
def apply_custom_css(theme):
    if theme == "Light":
        primary_color = "#ffffff"
        text_color = "#000000"
        bubble_user_bg = "#DCF8C6"  # Light green for user bubbles
        bubble_bot_bg = "#F1F0F0"  # Light grey for bot bubbles
        button_color = "#007bff"  # Blue for buttons
        button_hover_color = "#0056b3"  # Darker blue on hover
        border_color = "#ccc"
        shadow_color = "rgba(0, 0, 0, 0.1)"
    else:  # Dark theme
        primary_color = "#121212"
        text_color = "#ffffff"
        bubble_user_bg = "#1C1C1C"  # Dark for user bubbles
        bubble_bot_bg = "#2A2A2A"  # Dark grey for bot bubbles
        button_color = "#007bff"  # Blue for buttons
        button_hover_color = "#0056b3"  # Darker blue on hover
        border_color = "#6C757D"
        shadow_color = "rgba(255, 255, 255, 0.1)"

    st.markdown(f"""
    <style>
    body {{
        background-color: {primary_color};
        color: {text_color};
        font-family: 'Arial', sans-serif;
    }}
    .stTextInput > div > input {{
        padding: 10px;
        border-radius: 10px;
        border: 1px solid {border_color};
        background-color: {primary_color};
        color: {text_color};
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 12px {shadow_color};
    }}
    .stTextInput > div > input:hover {{
        box-shadow: 0px 6px 16px {shadow_color};
    }}
    .arrow-button {{
        background-color: {button_color};
        color: white;
        font-size: 18px;
        border-radius: 5px;
        border: none;
        padding: 10px 15px;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.2s ease;
        box-shadow: 0px 4px 12px {shadow_color};
    }}
    .arrow-button:hover {{
        background-color: {button_hover_color};
        box-shadow: 0px 6px 16px {shadow_color};
    }}
    .chat-bubble {{
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 12px;
        max-width: 75%;
        box-shadow: 0px 4px 12px {shadow_color};
        transition: transform 0.3s ease;
    }}
    .chat-bubble-user {{
        background-color: {bubble_user_bg};
        align-self: flex-end;
        color: {text_color};
        border: 1px solid {border_color};
    }}
    .chat-bubble-bot {{
        background-color: {bubble_bot_bg};
        align-self: flex-start;
        color: {text_color};
        border: 1px solid {border_color};
    }}
    .chat-bubble:hover {{
        transform: scale(1.02);
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
        width: 45px;
        height: 45px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0px 4px 8px {shadow_color};
    }}
    .title-text {{
        font-size: 40px;
        font-weight: bold;
        color: {text_color};
        text-shadow: 2px 2px 4px {shadow_color};
    }}
    .clear-chat-button {{
        background-color: #ff5c5c;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 15px;
        margin-top: 10px;
        cursor: pointer;
        box-shadow: 0px 4px 12px {shadow_color};
        transition: background-color 0.3s ease, box-shadow 0.2s ease;
    }}
    .clear-chat-button:hover {{
        background-color: #e04444;
        box-shadow: 0px 6px 16px {shadow_color};
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS based on theme
apply_custom_css(theme)

# Display the title and subtitle
st.markdown(f"<h1 class='title-text'>💬 Persona-based Conversational Bot</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle-text'>Ask me anything about health, finance, or personal development!</p>", unsafe_allow_html=True)

# Input box and submit button in a single row layout
col1, col2 = st.columns([4, 1])  # Adjust column ratios as needed
with col1:
    user_query = st.text_input("", placeholder="💡 Type your question here", key="user_input", label_visibility="collapsed")

with col2:
    if st.button("➡️", key="submit_button", type="primary"):
        if user_query.strip() != "":
            bot_response = generate_response(user_query)
        else:
            st.error("Please enter a valid query.")

# Chat window display
if len(st.session_state.chat_history) > 0:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for chat in st.session_state.chat_history:
        # User's chat bubble
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end;'>
            <div class='chat-bubble chat-bubble-user'>
                <img class='avatar' src='data:image/png;base64,{user_avatar_base64}' alt='User'>
                <span>{chat['user']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot's chat bubble
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-start;'>
            <div class='chat-bubble chat-bubble-bot'>
                <img class='avatar' src='data:image/png;base64,{bot_avatar_base64}' alt='Bot'>
                <span>{chat['bot']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Download chat history as .txt file
if st.button("💾 Download Chat History"):
    if len(st.session_state.chat_history) > 0:
        chat_content = "\n\n".join([f"User: {chat['user']}\nBot: {chat['bot']}" for chat in st.session_state.chat_history])
        b64 = base64.b64encode(chat_content.encode()).decode()  # Encode the content
        href = f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download Chat History</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No chat history available to download.")
