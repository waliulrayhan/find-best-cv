from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import fitz  # PyMuPDF
from docx import Document
from text_processor import preprocess_text, compute_similarity

app = FastAPI()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text based on file type"""
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

@app.post("/match-cvs")
async def match_cvs(
    job_description_file: UploadFile = File(...),
    cv_files: List[UploadFile] = File(...)
):
    """
    Upload job description (PDF/DOCX) and CVs (PDF/DOCX) to find best matches
    
    Args:
        job_description_file: Job description file (PDF or DOCX)
        cv_files: List of CV files (PDF or DOCX)
    
    Returns:
        Ranked list of CVs with similarity scores
    """
    # Create temporary directory for uploads
    upload_directory = "uploads"
    os.makedirs(upload_directory, exist_ok=True)

    try:
        # Process job description file
        jd_path = os.path.join(upload_directory, job_description_file.filename)
        with open(jd_path, "wb") as buffer:
            buffer.write(await job_description_file.read())
        
        # Extract and preprocess job description text
        try:
            jd_text = extract_text_from_file(jd_path, job_description_file.filename)
            processed_jd = preprocess_text(jd_text)
        finally:
            # Clean up job description file
            os.remove(jd_path)

        # Process CV files
        cv_texts = []
        cv_filenames = []
        processed_cvs = []

        for cv_file in cv_files:
            cv_path = os.path.join(upload_directory, cv_file.filename)
            
            try:
                # Save uploaded file
                with open(cv_path, "wb") as buffer:
                    buffer.write(await cv_file.read())
                
                # Extract and preprocess CV text
                cv_text = extract_text_from_file(cv_path, cv_file.filename)
                processed_cv = preprocess_text(cv_text)
                
                cv_texts.append(cv_text)
                processed_cvs.append(processed_cv)
                cv_filenames.append(cv_file.filename)
                
            except Exception as e:
                print(f"Error processing {cv_file.filename}: {str(e)}")
                continue
            finally:
                # Clean up CV file
                if os.path.exists(cv_path):
                    os.remove(cv_path)
        
        if not cv_texts:
            return {"error": "No valid CV files were uploaded"}
        
        # Compute similarities and rank CVs
        rankings = compute_similarity(jd_text, cv_texts)
        
        # Format response with detailed ranking information
        ranked_results = []
        for idx, score in rankings:
            ranked_results.append({
                "filename": cv_filenames[idx],
                "similarity_score": round(score, 2),
                "cv_preview": cv_texts[idx][:200] + "...",  # Preview of CV text
                "matched_keywords": get_matching_keywords(processed_jd, processed_cvs[idx])
            })
        
        return {
            "job_description": {
                "filename": job_description_file.filename,
                "preview": jd_text[:200] + "..."
            },
            "total_cvs_processed": len(cv_texts),
            "rankings": ranked_results
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def get_matching_keywords(jd_text: str, cv_text: str, top_n: int = 5) -> List[str]:
    """Extract matching keywords between job description and CV"""
    # Split texts into word sets
    jd_words = set(jd_text.split())
    cv_words = set(cv_text.split())
    
    # Find matching words
    matching_words = jd_words.intersection(cv_words)
    
    # Return top N matching words
    return list(matching_words)[:top_n]
