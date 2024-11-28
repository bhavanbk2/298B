import os
from abc import ABC, abstractmethod
from langchain_openai.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import streamlit as st

class ModelHandler(ABC):
    @abstractmethod
    def generate_response(self, query, persona, chat_history):
        pass

class GPTHandler(ModelHandler):
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
    
    def generate_response(self, query, persona, chat_history):
        try:
            context = " ".join([
                f"User: {item['user']} Bot: {item['bot']}" 
                for item in chat_history[-3:]  # Last 3 interactions
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
        self.authenticate()
        self.initialize_model()
        self.initialize_rag()
    
    def authenticate(self):
        try:
            self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
            self.cohere_api_key = os.getenv("COHERE_API_KEY")
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            
            if not all([self.hf_token, self.cohere_api_key, self.pinecone_api_key]):
                st.error("One or more required API tokens not found!")
                return False
            return True
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return False
    
    def initialize_model(self):
        try:
            if 'llama_model' not in st.session_state:
                max_seq_length = 2048
                dtype = None  # Auto detection
                load_in_4bit = True

                model_and_tokenizer = FastLanguageModel.from_pretrained(
                    model_name="shashikumar1998/Llama-3.2-3B-Instruct",
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    token=self.hf_token
                )

                if isinstance(model_and_tokenizer, tuple):
                    st.session_state.llama_model = model_and_tokenizer[0]
                    st.session_state.llama_tokenizer = model_and_tokenizer[1]
                else:
                    st.session_state.llama_model = model_and_tokenizer
                    
        except Exception as e:
            st.error(f"Error initializing Llama model: {str(e)}")
    
    def initialize_rag(self):
        try:
            if 'cohere_client' not in st.session_state:
                st.session_state.cohere_client = cohere.Client(self.cohere_api_key)
            
            if 'pinecone_index' not in st.session_state:
                pc = Pinecone(api_key=self.pinecone_api_key)
                st.session_state.pinecone_index = pc.Index("cohere-pinecone-tree")
        except Exception as e:
            st.error(f"Error initializing RAG components: {str(e)}")
    
    def generate_response(self, query, persona, chat_history):
        try:
            if not hasattr(st.session_state, 'llama_model'):
                return "Model not initialized. Please try again."
            
            # Get context from recent chat history
            context = " ".join([
                f"User: {item['user']} Assistant: {item['bot']}" 
                for item in chat_history[-3:]
            ])
            
            # Generate query embedding and retrieve relevant documents
            response = st.session_state.cohere_client.embed(
                texts=[query], 
                model="embed-english-light-v2.0"
            )
            query_embedding = response.embeddings[0]
            
            results = st.session_state.pinecone_index.query(
                vector=query_embedding, 
                top_k=5, 
                include_metadata=True
            )
            retrieved_context = "\n".join([
                result["metadata"]["text"] 
                for result in results["matches"]
            ])
            
            # Prepare messages with persona and context
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a {persona}. Here is some context: {retrieved_context}"
                },
                {"role": "user", "content": f"{context}\n{query}"}
            ]
            
            # Generate response
            inputs = st.session_state.llama_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            text_streamer = TextStreamer(
                st.session_state.llama_tokenizer, 
                skip_prompt=True
            )
            
            with st.spinner("Generating response..."):
                outputs = st.session_state.llama_model.generate(
                    input_ids=inputs,
                    streamer=text_streamer,
                    use_cache=True,
                    temperature=0.7,
                    min_p=0.1,
                    max_new_tokens=512
                )
            
            return st.session_state.llama_tokenizer.batch_decode(outputs)[0]
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Please try again."
