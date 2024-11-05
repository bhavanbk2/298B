import embedding as emb
import cohere
import os
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#create the index
#emb.prereq("cohere-pinecone-tree")

#Use the created index
index = emb.get_index("cohere-pinecone-tree")

#Cohere embedding 
co = cohere.Client(os.getenv("COHERE_API_KEY"))
#Query Search
query='What is the treatment for Salicylates?'

#Query embedding
q=co.embed(texts=[query],input_type="search_query",model="embed-english-v3.0").embeddings

#print(np.array(q).shape)

#query returning top 10 most similar results
result = index.query(vector=q, top_k=3,include_metadata=True)

results = ""
for match in result['matches']:
    results = results + match['metadata']['text']
    #print(f"{match['score']:.2f}: {match['metadata']['text']}")
    
model = AutoModelForCausalLM.from_pretrained("meta-llama-2-7b-chat-hf",cache_dir="/data/chatbot/base_models")