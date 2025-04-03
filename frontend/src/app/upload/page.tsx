"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit

export default function Upload() {
  const router = useRouter();
  const [jobFile, setJobFile] = useState<File | null>(null);
  const [cvFiles, setCvFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Fix hydration mismatch by only rendering after component mounts
  useEffect(() => {
    setMounted(true);
  }, []);

  // Skip rendering until after mount
  if (!mounted) {
    return <LoadingState message="Loading..." />;
  }

  const validateFile = (file: File) => {
    if (file.size > MAX_FILE_SIZE) {
      return `File ${file.name} is too large. Maximum size is 10MB.`;
    }
    
    const validTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    
    if (!validTypes.includes(file.type)) {
      return `File ${file.name} is not a valid format. Please use PDF or DOCX.`;
    }
    
    return null;
  };

  const handleJobFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const error = validateFile(file);
      if (error) {
        setError(error);
        return;
      }
      setJobFile(file);
      setError(null);
      setSuccess("Job description file uploaded successfully!");
    }
  };

  const handleCvFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      
      // Validate each file
      for (const file of files) {
        const error = validateFile(file);
        if (error) {
          setError(error);
          return;
        }
      }
      
      setCvFiles(files);
      setError(null);
      setSuccess(`${files.length} CV file(s) uploaded successfully!`);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!jobFile) {
      setError("Please upload a job description file");
      return;
    }
    
    if (cvFiles.length === 0) {
      setError("Please upload at least one CV file");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    
    const formData = new FormData();
    formData.append("job_description_file", jobFile);
    
    cvFiles.forEach(file => {
      formData.append("cv_files", file);
    });
    
    try {
      // Add error handling and logging to debug the issue
      console.log("Sending request to backend...");
      
      const response = await fetch("http://127.0.0.1:8000/match-cvs", {
        method: "POST",
        body: formData,
        // Add these headers to help with CORS
        mode: 'cors',
        credentials: 'same-origin',
      });
      
      console.log("Response received:", response);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        throw new Error(`Error ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      console.log("Data received:", data);
      
      // Store results in localStorage to pass to results page
      localStorage.setItem("matchResults", JSON.stringify(data));
      
      // Navigate to results page
      router.push("/results");
    } catch (err) {
      console.error("Fetch error:", err);
      setError(`Failed to process files: ${err instanceof Error ? err.message : String(err)}`);
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-blue-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-800 tracking-tight">
            Upload Your Files
          </h1>
          <p className="mt-3 text-lg text-gray-600">
            Upload your job description and CV files to find the perfect match
          </p>
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 shadow-sm animate-fadeIn">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {success && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-4 mb-6 shadow-sm animate-fadeIn">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800">{success}</p>
              </div>
            </div>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl">
            <div className="px-6 py-6 sm:p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                Job Description
              </h3>
              <div className="h-1 w-16 bg-blue-600 rounded-full mb-4"></div>
              <div className="text-sm text-gray-600 mb-5">
                Upload your job description file (PDF or DOCX)
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-center w-full">
                  <label className="flex flex-col w-full h-40 border-2 border-dashed border-blue-200 rounded-xl cursor-pointer hover:bg-blue-50 transition-colors duration-300">
                    <div className="flex flex-col items-center justify-center pt-7">
                      {jobFile ? (
                        <>
                          <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-2">
                            <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                          </div>
                          <p className="text-sm font-medium text-gray-700">
                            {jobFile.name}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Click to change file
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mb-2">
                            <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                          </div>
                          <p className="text-sm font-medium text-gray-700">
                            Click to upload job description
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            PDF or DOCX (max 10MB)
                          </p>
                        </>
                      )}
                    </div>
                    <input 
                      type="file" 
                      className="opacity-0" 
                      accept=".pdf,.docx" 
                      onChange={handleJobFileChange}
                    />
                  </label>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl">
            <div className="px-6 py-6 sm:p-8">
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                CV Files
              </h3>
              <div className="h-1 w-16 bg-blue-600 rounded-full mb-4"></div>
              <div className="text-sm text-gray-600 mb-5">
                Upload multiple CV files (PDF or DOCX)
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-center w-full">
                  <label className="flex flex-col w-full h-40 border-2 border-dashed border-blue-200 rounded-xl cursor-pointer hover:bg-blue-50 transition-colors duration-300">
                    <div className="flex flex-col items-center justify-center pt-7">
                      {cvFiles.length > 0 ? (
                        <>
                          <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-2">
                            <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                          </div>
                          <p className="text-sm font-medium text-gray-700">
                            {cvFiles.length} file(s) selected
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Click to change files
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mb-2">
                            <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                          </div>
                          <p className="text-sm font-medium text-gray-700">
                            Click to upload CV files
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            PDF or DOCX (max 10MB each)
                          </p>
                        </>
                      )}
                    </div>
                    <input 
                      type="file" 
                      className="opacity-0" 
                      accept=".pdf,.docx" 
                      multiple 
                      onChange={handleCvFilesChange}
                    />
                  </label>
                </div>
              </div>
              {cvFiles.length > 0 && (
                <div className="mt-6 bg-gray-50 rounded-xl p-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Selected files:</h4>
                  <div className="max-h-40 overflow-y-auto pr-2">
                    <ul className="divide-y divide-gray-200">
                      {cvFiles.map((file, index) => (
                        <li key={index} className="py-2 flex items-center">
                          <svg className="h-4 w-4 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                          </svg>
                          <span className="text-sm text-gray-600">{file.name}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="flex justify-center mt-10">
            <button
              type="submit"
              disabled={isLoading}
              className={`px-8 py-4 text-base font-medium rounded-full bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                'Match CVs'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
} 