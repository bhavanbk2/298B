import os
from dotenv import load_dotenv
import cohere
import data_preprocess as dp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import time
import math
load_dotenv()
def prereq(index_name):
     # Load environment variables from .env file
     co = cohere.Client(os.getenv("COHERE_API_KEY"))

     # Define base directory where PDFs are stored
     base_dir = "C:/Project_Data"

     #Get the domain data and save in .txt file format after cleansing
     dp.process_domain(base_dir)
     print("Data has been cleansed and stored in .txt format")

     # We use a hierarchical list of separators specifically tailored for splitting Markdown documents
     # This list is taken from LangChain's MarkdownTextSplitter class.
     MARKDOWN_SEPARATORS = [
     "\n#{1,6} ",
     "```\n",
     "\n\\*\\*\\*+\n",
     "\n---+\n",
     "\n___+\n",
     "\n\n",
     "\n",
     " ",
     "",
     ]

     text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=1024,  # the maximum number of characters in a chunk: we selected this value arbitrarily
     chunk_overlap=24,  # the number of characters to overlap between chunks
     #add_start_index=True,  # If `True`, includes chunk's start index in metadata
     #strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
     separators=MARKDOWN_SEPARATORS,
     )

     #Load the cleansed domain data stored in .txt file
     loaderMedical = TextLoader(base_dir+'/cleansed_data/domain.txt',encoding='utf-8')
     Medicaldocument = loaderMedical.load()
     docs_processed = []
     docs_processed += text_splitter.split_documents(Medicaldocument)
     '''
     loaderFinance= TextLoader(base_dir+'/cleansed_data/Finance.txt',encoding='utf-8')
     Financedocument = loaderFinance.load()
     docs_processed += text_splitter.split_documents(Financedocument)
     '''
     #Add the chunked documents into a list for embedding
     data = []
     for i in range(len(docs_processed)):
          data.append(docs_processed[i].page_content)
     print("Number of chunks: ",len(data))
     print("Starting Vector embedding now......")

     #Encode your documents with input type 'search_document'
     for i in range(math.ceil(len(data)/100)):
          doc_emb = co.embed(data[0 if i==0 else (i*100)-1:min(((i+1)*100),len(data))], input_type="search_document", model="embed-english-v3.0").embeddings
          print("start index:",0 if i==0 else (i*100)-1)
          print("End index:",min(((i+1)*100),len(data)))
          shape = np.asarray(doc_emb).shape
          vector_db(index_name,doc_emb,shape,data[0 if i==0 else (i*100)-1:min(((i+1)*100),len(data))])
          #Sleep since we can't call more than 100 api calls per min
          print("sleep starting")
          time.sleep(65)
 
     

def get_index(index_name):     
     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
     index = pc.Index(index_name)
     return index

def vector_db(index_name,doc_emb,shape,data):
     #doc_emb,shape,data = prereq()
     #Pinecone to store the vector embeddings
     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
     #index_name='cohere-pinecone-tree'
     #delete index if exists
     #if index_name not in pc.list_indexes().names():
     #     pc.delete_index(index_name)
     #Recreate index
     if index_name not in pc.list_indexes().names():
          pc.create_index(index_name, dimension=shape[1],metric='cosine',spec=ServerlessSpec(cloud='aws', region='us-west-2')
     )

     index = pc.Index(index_name)

     batch_size = 128

     ids = [str(i) for i in range(shape[0])]

     #List of metadata dictionaries
     meta = [{'text':text} for text in data]

     #List of (id,domain,vector,metadata) tuples to be upserted
     to_upsert=list(zip(ids,doc_emb,meta))

     for i in range(0,shape[0],batch_size):
          i_end=min(i+batch_size,shape[0])
          index.upsert(vectors=to_upsert[i:i_end])
     print("Vector db successfuly created in pinecone")