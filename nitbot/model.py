from PyPDF2 import PdfReader

import dotenv
import os
dotenv.load_dotenv()
from langchain_core.prompts import PromptTemplate

api_key = os.environ.get("GROQ_API_KEY")
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") 
from .EMB import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain.chains.question_answering import load_qa_chain as original_load_qa_chain
# from groq import Groq
from langchain_groq import ChatGroq
llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile",  # You can choose different models
    temperature=0,
    max_tokens=1024
)

# Export the load_qa_chain function to be used in index.py
def load_qa_chain(llm_model):
    
    prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are a helpful assistant with access to document information and conversation memory."

        "**Document Information:** {context}"

        "**Conversation History:** {chat_history}"

        "**User's Current Question:** {question}"

        "**Instructions:**"
        "- Always maintain conversational context from previous exchanges in the chat history."
        "- For follow-up questions, refer to the conversation history to understand references like 'it', 'this', 'that', or other pronouns. If the reference is unclear even after checking the history, politely ask the user for clarification."
        "- Respond naturally, as if you remember the entire conversation, using a friendly and engaging tone."
        "- Only provide answers based on the document information or conversation history. Do not generate or infer information beyond what is explicitly provided."
        "- If the answer to a question is not found in the document information or conversation history, respond exactly with: \"I can only answer questions about uploaded documents.\""
        "- If the user asks about topics outside the scope of the provided documents or chat history, respond only with: \"I can only answer questions about uploaded documents.\""
        "- Never fabricate, hallucinate, or assume information not present in the documents or conversation history."
        "- Ensure responses are concise yet complete, avoiding unnecessary elaboration unless requested by the user."

        "**Answer conversationally:**"
        
        
        
        # "You are a helpful assistant with access to document information and conversation memory.\n\n"
        # "Document Information: {context}\n\n"
        # "Conversation History: {chat_history}\n\n"
        # "User's Current Question: {question}\n\n"
        # "Instructions:\n"
        # "- Maintain conversational context from previous exchanges\n"
        # "- For follow-up questions, remember what was previously discussed\n"
        # "- If the user asks about 'it', 'this', 'that', or uses other references, determine what they're referring to from context\n"
        # "- Answer naturally as if you remember the entire conversation\n"
        # "- If it wasn't clear who I was referring to, first check chat history and even still not clear ask for clarification.\n"
        # "- Only use information found in the document information or directly related to previous conversation\n"
        # "- If the answer is not found in the document information or conversation history, respond only with: \"I can only answer questions about uploaded documents.\"\n"
        # "- Never make up or hallucinate information not present in the documents\n"
        # "- If asked about topics outside the scope of uploaded documents, respond with: \"I can only answer questions about uploaded documents.\"\n\n"
        # "Answer conversationally:"
    )
)
    
    
    return create_stuff_documents_chain(llm=llm_model,
                                        prompt=prompt,
                                        document_variable_name="context",)