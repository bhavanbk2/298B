# 298B
# SageSoulRAG: RAG-based Chatbot with Personality

Aradhya A. Rathnakar, Bhavan K. Basavaraju, Mahamaya Panda, ReddySaketh R. Chappidi, and ShashiKumar K. Mallikarjuna

_Department of Applied Data Science, San Jose State University
DATA 298B: MSDA Project II
Simon Shim
May 13, 2024_

# Abstract #
Conversational AI is making impressive strides, but the current versions of chatbots tend to give responses that feel canned and impersonal. This project introduces a cutting-edge, RAG-based conversational AI that harnesses the personas of finance and medical experts, Mr. Robert Kiyosaki and Dr. Sanjay Gupta. By utilizing four advanced large language models (LLMs)—GPT-3.5 Turbo, Mistral 7B, Llama 2, and PaLM 2—integrated through an API, this AI application provides nuanced, expert-level advice tailored to specific domains. The system's architecture includes a comprehensive data pipeline featuring a knowledge base in both the finance and healthcare sectors. The data preprocessing tasks such as cleansing, tokenization, normalization, and cohere embedding are applied before storing the data in the Pinecone Vector Database for optimized retrieval. User inputs are similarly preprocessed to enhance the matching process within the vector database. Central to the approach is the fine-tuning of these LLMs with persona-specific datasets representing Mr.Kiyosaki and Dr.Gupta. This fine-tuning process ensures that the chatbot not only delivers responses that are accurate and relevant to the specific domain but also reflects the distinct communication styles and expert insights of these figures. The interface is facilitated through a user-friendly chatbot UI that leverages these fine-tuned models to produce responses that are engaging and informative. The performance of the conversational AI is rigorously evaluated using metrics such as OpenAI Eval and DeepEval for response fluency and relevance, which includes metrics such as Bootstrap Accuracy Standard Deviation, hallucinations, answer relevancy, contextual Precision, Recall, and relevancy to measure how well the system aligns with user intents. This RAG-based chatbot aims to be the best-in-class tool for providing engaging, personality-driven advice in finance and healthcare, making it a transformative tool for users seeking expert guidance.
