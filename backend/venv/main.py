from fastapi import FastAPI, File, UploadFile
from typing import List
import os

app = FastAPI()

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

    for file in files:
        file_path = os.path.join(upload_directory, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

    return {"message": "Files uploaded successfully"}
