from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Tuple
import os
import fitz  # PyMuPDF
from docx import Document
from text_processor import preprocess_text, compute_similarity
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing DOCX: {str(e)}")

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text based on file type"""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

@app.get("/")
async def read_root():
    return {"status": "API is running"}

@app.post("/match-cvs")
async def match_cvs(
    job_description_file: UploadFile = File(...),
    cv_files: List[UploadFile] = File(...)
):
    """
    Upload job description (PDF/DOCX) and CVs (PDF/DOCX) to find best matches
    """
    logger.info(f"Received request with job description: {job_description_file.filename} and {len(cv_files)} CVs")
    
    # Create temporary directory for uploads
    upload_directory = "uploads"
    os.makedirs(upload_directory, exist_ok=True)

    try:
        # Process job description file
        jd_path = os.path.join(upload_directory, job_description_file.filename)
        with open(jd_path, "wb") as buffer:
            buffer.write(await job_description_file.read())
        
        logger.info(f"Saved job description file: {jd_path}")
        
        # Extract and preprocess job description text
        try:
            jd_text = extract_text_from_file(jd_path, job_description_file.filename)
            processed_jd = preprocess_text(jd_text)
            logger.info(f"Processed job description: {job_description_file.filename}")
        except Exception as e:
            logger.error(f"Error processing job description: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing job description: {str(e)}")
        finally:
            # Clean up job description file
            if os.path.exists(jd_path):
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
                
                logger.info(f"Saved CV file: {cv_path}")
                
                # Extract and preprocess CV text
                cv_text = extract_text_from_file(cv_path, cv_file.filename)
                processed_cv = preprocess_text(cv_text)
                
                cv_texts.append(cv_text)
                processed_cvs.append(processed_cv)
                cv_filenames.append(cv_file.filename)
                
                logger.info(f"Processed CV: {cv_file.filename}")
                
            except Exception as e:
                logger.error(f"Error processing {cv_file.filename}: {str(e)}")
                continue
            finally:
                # Clean up CV file
                if os.path.exists(cv_path):
                    os.remove(cv_path)
        
        if not cv_texts:
            logger.warning("No valid CV files were uploaded")
            raise HTTPException(status_code=400, detail="No valid CV files were uploaded")
        
        # Compute similarities and rank CVs
        rankings = compute_similarity(jd_text, cv_texts)
        
        # Format response with detailed ranking information
        ranked_results = []
        for idx, score in rankings:
            # Get matching keywords (simplified version without mapping)
            matched_keywords = get_matching_keywords(processed_jd, processed_cvs[idx])
            
            ranked_results.append({
                "filename": cv_filenames[idx],
                "similarity_score": round(score, 2),
                "cv_preview": cv_texts[idx][:200] + "...",  # Preview of CV text
                "matched_keywords": matched_keywords
            })
        
        logger.info(f"Completed ranking {len(cv_texts)} CVs")
        
        response_data = {
            "job_description": {
                "filename": job_description_file.filename,
                "preview": jd_text[:200] + "..."
            },
            "total_cvs_processed": len(cv_texts),
            "rankings": ranked_results
        }
        
        logger.info("Returning response")
        return response_data

    except Exception as e:
        logger.error(f"Error in match-cvs endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def get_matching_keywords(jd_text: str, cv_text: str, top_n: int = 5) -> List[str]:
    """Extract matching keywords between job description and CV"""
    # Split texts into word sets
    jd_words = set(jd_text.split())
    cv_words = set(cv_text.split())
    
    # Find matching words
    matching_words = jd_words.intersection(cv_words)
    
    # Return top N matching words
    return list(matching_words)[:top_n]

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
