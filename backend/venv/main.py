from fastapi import FastAPI, File, UploadFile, Body
from typing import List, Dict
import os
import fitz  # PyMuPDF
from docx import Document
from text_processor import preprocess_text, vectorize_text

app = FastAPI()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/health")
async def health_check():
    return {"status": "running"}

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    upload_directory = "uploads"
    os.makedirs(upload_directory, exist_ok=True)

    extracted_texts = {}
    processed_texts = {}

    for file in files:
        file_path = os.path.join(upload_directory, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
        elif file.filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_path)
        else:
            extracted_text = "Unsupported file type"
            
        extracted_texts[file.filename] = extracted_text
        
        # Preprocess the extracted text
        if extracted_text != "Unsupported file type":
            processed_text = preprocess_text(extracted_text)
            processed_texts[file.filename] = processed_text

    return {
        "extracted_texts": extracted_texts,
        "processed_texts": processed_texts
    }

@app.post("/process-text")
async def process_text(text_data: Dict[str, str] = Body(...)):
    """
    Process CV text and return TF-IDF vector
    
    Request body:
    {
        "text": "CV text content here"
    }
    """
    raw_text = text_data.get("text", "")
    
    # Preprocess the text
    processed_text = preprocess_text(raw_text)
    
    # Vectorize the text
    tfidf_vector = vectorize_text(processed_text, fit_vectorizer=True)
    
    return {
        "processed_text": processed_text,
        "tfidf_vector": tfidf_vector
    }
