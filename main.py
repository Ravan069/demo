from fastapi import FastAPI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# FastAPI app initialization
app = FastAPI()

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# Function to extract text from a static PDF
def get_pdf_text(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

# Function to convert chunks into vector embeddings
def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Function to handle conversation
def conversation_chain():
    template = """
    Answer the asked question in detail.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt), model_instance

# Function to process the user's question
def user_question(question, db, chain, raw_text):
    if db is None:
        return "Please process the PDF first."
    docs = db.similarity_search(question, k=5)
    response = chain.invoke({"input_documents": docs, "question": question, "context": raw_text}, return_only_outputs=True)
    return response.get("output_text")

# Static PDF File Path
pdf_file_path = "c.pdf"

# Read and process the static PDF once
raw_text = get_pdf_text(pdf_file_path)
if raw_text:
    chunks = get_text_chunks(raw_text)
    vector_store = get_vector(chunks)
    chain, _ = conversation_chain()

# Endpoint to ask a question
@app.get("/ask/{question}")
async def ask_question(question: str):
    if not vector_store:
        return {"message": "Please process the PDF first."}
    response_text = user_question(question, vector_store, chain, raw_text)
    return {"answer": response_text}
