from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Dict
import os
import shutil
import logging
import uuid
from text_processor import preprocess_text, compute_similarity, get_matching_keywords

# Import scikit-learn components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import PyMuPDF and python-docx with proper error handling
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is not installed. Run 'pip install pymupdf'")

try:
    from docx import Document
except ImportError:
    raise ImportError("python-docx is not installed. Run 'pip install python-docx'")

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
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store file mappings
file_mappings: Dict[str, str] = {}

def save_upload_file(upload_file: UploadFile) -> str:
    """Save an upload file to disk and return the path"""
    # Create a unique filename to avoid collisions
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Reset the file pointer for future reads
    upload_file.file.seek(0)
    
    # Store the mapping between original filename and saved path
    file_mappings[upload_file.filename] = file_path
    
    return file_path

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

@app.post("/match-cvs")
async def match_cvs(
    job_description_file: UploadFile = File(...),
    cv_files: List[UploadFile] = File(...)
):
    """
    Upload job description (PDF/DOCX) and CVs (PDF/DOCX) to find best matches
    """
    jd_path = None
    cv_paths = []
    
    try:
        logger.info(f"Processing job description: {job_description_file.filename}")
        
        # Save job description file
        jd_path = save_upload_file(job_description_file)
        
        # Extract text from job description
        jd_text = extract_text_from_file(jd_path, job_description_file.filename)
        processed_jd = preprocess_text(jd_text)
        
        # Process CV files
        cv_texts = []
        processed_cvs = []
        cv_filenames = []
        
        for cv_file in cv_files:
            try:
                logger.info(f"Processing CV: {cv_file.filename}")
                
                # Save CV file
                cv_path = save_upload_file(cv_file)
                cv_paths.append(cv_path)
                
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
        
        if not cv_texts:
            logger.warning("No valid CV files were uploaded")
            raise HTTPException(status_code=400, detail="No valid CV files were uploaded")
        
        # Compute similarities and rank CVs
        rankings = compute_similarity(jd_text, cv_texts)
        
        # Format response with detailed ranking information
        ranked_results = []
        for idx, score in rankings:
            # Get matching keywords
            matched_keywords = get_matching_keywords(processed_jd, processed_cvs[idx])
            
            # Log the score for debugging
            logger.info(f"CV: {cv_filenames[idx]}, Score: {score}")
            
            ranked_results.append({
                "filename": cv_filenames[idx],
                "similarity_score": round(score, 2),  # Round to 2 decimal places
                "cv_preview": cv_texts[idx][:200] + "...",
                "full_text": cv_texts[idx],
                "matched_keywords": matched_keywords
            })
        
        logger.info(f"Completed ranking {len(cv_texts)} CVs")
        
        response_data = {
            "job_description": {
                "filename": job_description_file.filename,
                "preview": jd_text[:200] + "...",
                "full_text": jd_text
            },
            "total_cvs_processed": len(cv_texts),
            "rankings": ranked_results
        }
        
        logger.info("Returning response")
        return response_data

    except Exception as e:
        logger.error(f"Error in match-cvs endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        # Keep files for download
        pass

@app.get("/download-file/{filename}")
async def download_file(filename: str):
    """Download a previously uploaded file by its original filename"""
    if filename not in file_mappings:
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    file_path = file_mappings[filename]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} no longer exists on the server")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
