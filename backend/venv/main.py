from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import fitz  # PyMuPDF
from docx import Document

app = FastAPI()

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

    for file in files:
        file_path = os.path.join(upload_directory, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if file.filename.endswith(".pdf"):
            extracted_texts[file.filename] = extract_text_from_pdf(file_path)
        elif file.filename.endswith(".docx"):
            extracted_texts[file.filename] = extract_text_from_docx(file_path)
        else:
            extracted_texts[file.filename] = "Unsupported file type"

    return {"extracted_texts": extracted_texts}
