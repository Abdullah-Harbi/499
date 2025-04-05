import os
import json
from pathlib import Path
from pprint import pprint
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from uuid import uuid4
from langchain.chains import ConversationalRetrievalChain

# Configure Gemini API
import google.generativeai as genai

# Set your Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Make sure to set this in your environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it before running the script.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize generative AI model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# Ensure the directory path is absolute and exists
pdf_directory = Path("pdfs").resolve()

if not pdf_directory.exists():
    raise FileNotFoundError(f"Directory not found: {pdf_directory}")

pdf_files = list(pdf_directory.rglob("*.pdf"))  # Use rglob for recursive search
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in directory: {pdf_directory}")

print("Found PDF files:")
for pdf in pdf_files:
    print(pdf)

# Load all pages from all PDFs
all_pages = []  # Store all pages from all PDFs

for pdf_path in pdf_files:
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(str(pdf_path))
    pdf_pages = loader.load()  # Load pages
    all_pages.extend(pdf_pages)  # Append pages to the list

print(f"Total pages loaded: {len(all_pages)}")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(all_pages)

# Embed and vectorize text
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vector_store = Chroma.from_documents(chunks, embeddings)

# Set conditional prompt
condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

qa_template = """Do not attempt to generate answers from an external source.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Answer:"""
QA_PROMPT = PromptTemplate.from_template(qa_template)

# In-memory session storage
sessions = {}

# Function to create a new session
def create_session():
    session_id = str(uuid4())  # Generate a unique session ID
    # Specify output_key for memory
    sessions[session_id] = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"  # Explicitly set the output key
    )
    return session_id

# Function to retrieve or create session memory
def get_session_memory(session_id):
    if session_id not in sessions:
        # Specify output_key for memory
        sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"  # Explicitly set the output key
        )
    return sessions[session_id]

# Function to terminate a session
def terminate_session(session_id):
    if session_id in sessions:
        del sessions[session_id]

# Initialize ConversationalRetrievalChain with session memory
def get_chain(session_id):
    memory = get_session_memory(session_id)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

# Query the model with session memory
def chat(session_id, user_question):
    qa_chain = get_chain(session_id)
    
    # Invoke the chain with the question
    result = qa_chain({"question": user_question})
    
    # Print source documents if available (optional)
    if "source_documents" in result:
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
    
    return result["answer"]

# Main loop
if __name__ == "__main__":
    # Step 1: Create a new session
    session_id = create_session()
    print(f"Session ID: {session_id}")

    while True:
        user_question = input("Input your question (or type 'exit' to end): ")
        if user_question.lower() == "exit":
            break

        try:
            # Step 2: Chat with the bot
            response = chat(session_id, user_question)
            pprint(response)
        except Exception as e:
            print(f"Error: {e}")

    # Step 3: Terminate the session
    terminate_session(session_id)
    print(f"Session {session_id} terminated.")