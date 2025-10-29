"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";
import { motion } from "framer-motion";
import { useToast } from "../context/ToastContext";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit

export default function Upload() {
  const router = useRouter();
  const { showToast } = useToast();
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
      const errorMsg = validateFile(file);
      if (errorMsg) {
        setError(errorMsg);
        showToast(errorMsg, "error");
        return;
      }
      setJobFile(file);
      setError(null);
      const successMsg = "Job description file uploaded successfully!";
      setSuccess(successMsg);
      showToast(successMsg, "success");
    }
  };

  const handleCvFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      
      // Validate each file
      for (const file of files) {
        const errorMsg = validateFile(file);
        if (errorMsg) {
          setError(errorMsg);
          showToast(errorMsg, "error");
          return;
        }
      }
      
      setCvFiles(files);
      setError(null);
      const successMsg = `${files.length} CV file(s) uploaded successfully!`;
      setSuccess(successMsg);
      showToast(successMsg, "success");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!jobFile) {
      const errorMsg = "Please upload a job description file";
      setError(errorMsg);
      showToast(errorMsg, "error");
      return;
    }
    
    if (cvFiles.length === 0) {
      const errorMsg = "Please upload at least one CV file";
      setError(errorMsg);
      showToast(errorMsg, "error");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    showToast("Processing your files...", "info");
    
    const formData = new FormData();
    formData.append("job_description_file", jobFile);
    
    cvFiles.forEach(file => {
      formData.append("cv_files", file);
    });
    
    try {
      // Add error handling and logging to debug the issue
      console.log("Sending request to backend...");
      
      const response = await fetch(process.env.NEXT_PUBLIC_API_URL || "https://cv-matcher-api.onrender.com/match-cvs", {
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
      
      showToast("Files processed successfully!", "success");
      
      // Store results in localStorage to pass to results page
      localStorage.setItem("matchResults", JSON.stringify(data));
      
      // Navigate to results page
      router.push("/results");
    } catch (err) {
      console.error("Fetch error:", err);
      const errorMsg = `Failed to process files: ${err instanceof Error ? err.message : String(err)}`;
      setError(errorMsg);
      showToast(errorMsg, "error");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] py-12 px-4 sm:px-6 lg:px-8">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <div className="text-center mb-12">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="inline-block mb-4"
          >
            <div className="w-20 h-20 mx-auto bg-gradient-to-br from-[#1E3A8A] to-[#10B981] rounded-full flex items-center justify-center shadow-lg">
              <motion.svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-10 w-10 text-white" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
                animate={{ rotate: [0, 10, 0] }}
                transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </motion.svg>
            </div>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="text-4xl font-extrabold text-[#1E3A8A] tracking-tight"
          >
            Upload Your Files
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-3 text-lg text-[#374151]"
          >
            Upload your job description and CV files to find the perfect match
          </motion.p>
        </div>
        
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border-l-4 border-red-500 rounded-xl p-4 mb-6 shadow-sm"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <motion.svg 
                  animate={{ rotate: [0, 10, 0] }}
                  transition={{ repeat: 3, duration: 0.3 }}
                  className="h-5 w-5 text-red-400" 
                  viewBox="0 0 20 20" 
                  fill="currentColor"
                >
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </motion.svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
            </div>
          </motion.div>
        )}
        
        {success && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-50 border-l-4 border-[#10B981] rounded-xl p-4 mb-6 shadow-sm"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <motion.svg 
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: 1, duration: 0.5 }}
                  className="h-5 w-5 text-[#10B981]" 
                  viewBox="0 0 20 20" 
                  fill="currentColor"
                >
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </motion.svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800">{success}</p>
              </div>
            </div>
          </motion.div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="bg-white rounded-2xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl border border-gray-100"
          >
            <div className="px-6 py-6 sm:p-8">
              <div className="flex items-center mb-4">
                <motion.div
                  whileHover={{ rotate: 15 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  className="bg-[#1E3A8A] p-2 rounded-lg mr-3 shadow-md"
                >
                  <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                  </svg>
                </motion.div>
                <h3 className="text-xl font-bold text-[#1E3A8A]">
                  Job Description
                </h3>
              </div>
              <div className="h-1 w-24 bg-gradient-to-r from-[#1E3A8A] to-[#10B981] rounded-full mb-4"></div>
              <div className="text-sm text-[#374151] mb-5">
                Upload your job description file (PDF or DOCX)
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-center w-full">
                  <motion.label 
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="flex flex-col w-full h-40 border-2 border-dashed border-[#1E3A8A]/30 rounded-xl cursor-pointer hover:bg-blue-50 transition-colors duration-300"
                  >
                    <div className="flex flex-col items-center justify-center pt-7">
                      {jobFile ? (
                        <>
                          <motion.div 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 400, damping: 10 }}
                            className="w-16 h-16 rounded-full bg-[#10B981]/20 flex items-center justify-center mb-2"
                          >
                            <svg className="w-8 h-8 text-[#10B981]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                          </motion.div>
                          <p className="text-sm font-medium text-[#374151]">
                            {jobFile.name}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Click to change file
                          </p>
                        </>
                      ) : (
                        <>
                          <motion.div 
                            animate={{ y: [0, -5, 0] }}
                            transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                            className="w-16 h-16 rounded-full bg-[#1E3A8A]/10 flex items-center justify-center mb-2"
                          >
                            <svg className="w-8 h-8 text-[#1E3A8A]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                          </motion.div>
                          <p className="text-sm font-medium text-[#374151]">
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
                  </motion.label>
                </div>
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="bg-white rounded-2xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl border border-gray-100"
          >
            <div className="px-6 py-6 sm:p-8">
              <div className="flex items-center mb-4">
                <motion.div
                  whileHover={{ rotate: 15 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  className="bg-[#1E3A8A] p-2 rounded-lg mr-3 shadow-md"
                >
                  <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                  </svg>
                </motion.div>
                <h3 className="text-xl font-bold text-[#1E3A8A]">
                  CV Files
                </h3>
              </div>
              <div className="h-1 w-24 bg-gradient-to-r from-[#1E3A8A] to-[#10B981] rounded-full mb-4"></div>
              <div className="text-sm text-[#374151] mb-5">
                Upload multiple CV files (PDF or DOCX)
              </div>
              <div className="mt-4">
                <div className="flex items-center justify-center w-full">
                  <motion.label 
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    className="flex flex-col w-full h-40 border-2 border-dashed border-[#1E3A8A]/30 rounded-xl cursor-pointer hover:bg-blue-50 transition-colors duration-300"
                  >
                    <div className="flex flex-col items-center justify-center pt-7">
                      {cvFiles.length > 0 ? (
                        <>
                          <motion.div 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 400, damping: 10 }}
                            className="w-16 h-16 rounded-full bg-[#10B981]/20 flex items-center justify-center mb-2"
                          >
                            <svg className="w-8 h-8 text-[#10B981]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                          </motion.div>
                          <p className="text-sm font-medium text-[#374151]">
                            {cvFiles.length} file(s) selected
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Click to change files
                          </p>
                        </>
                      ) : (
                        <>
                          <motion.div 
                            animate={{ y: [0, -5, 0] }}
                            transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                            className="w-16 h-16 rounded-full bg-[#1E3A8A]/10 flex items-center justify-center mb-2"
                          >
                            <svg className="w-8 h-8 text-[#1E3A8A]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                          </motion.div>
                          <p className="text-sm font-medium text-[#374151]">
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
                  </motion.label>
                </div>
              </div>
              {cvFiles.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ duration: 0.3 }}
                  className="mt-6 bg-[#F8FAFC] rounded-xl p-4 border border-[#1E3A8A]/10"
                >
                  <h4 className="text-sm font-medium text-[#1E3A8A] mb-2 flex items-center">
                    <svg className="h-4 w-4 text-[#F59E0B] mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    Selected files:
                  </h4>
                  <div className="max-h-40 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-[#1E3A8A]/20 scrollbar-track-transparent">
                    <ul className="divide-y divide-[#1E3A8A]/10">
                      {cvFiles.map((file, index) => (
                        <motion.li 
                          key={index} 
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="py-2 flex items-center"
                        >
                          <svg className="h-4 w-4 text-[#1E3A8A] mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                          </svg>
                          <span className="text-sm text-[#374151]">{file.name}</span>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
            className="flex justify-center mt-10"
          >
            <motion.button
              type="submit"
              disabled={isLoading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`px-8 py-4 text-base font-medium rounded-full bg-gradient-to-r from-[#1E3A8A] to-[#10B981] text-white shadow-lg hover:shadow-xl transform transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#1E3A8A] ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
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
                <>
                  <span className="mr-2">Match CVs</span>
                  <svg className="w-5 h-5 inline-block" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                  </svg>
                </>
              )}
            </motion.button>
          </motion.div>
        </form>
      </motion.div>
    </div>
  );
} 