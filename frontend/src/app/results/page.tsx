"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";

type MatchResult = {
  job_description: {
    filename: string;
    preview: string;
    full_text: string;
  };
  total_cvs_processed: number;
  rankings: {
    filename: string;
    similarity_score: number;
    cv_preview: string;
    full_text: string;
    matched_keywords: string[];
  }[];
};

export default function Results() {
  const router = useRouter();
  const [results, setResults] = useState<MatchResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [showJobDescription, setShowJobDescription] = useState(false);
  const [selectedCV, setSelectedCV] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 5;

  useEffect(() => {
    setMounted(true);
    
    // Get results from localStorage
    const storedResults = localStorage.getItem("matchResults");
    
    if (storedResults) {
      setResults(JSON.parse(storedResults));
    } else {
      // If no results, redirect to upload page
      router.push("/upload");
    }
    
    setLoading(false);
  }, [router]);

  if (!mounted) {
    return <LoadingState message="Loading..." />;
  }

  if (loading) {
    return <LoadingState message="Loading results..." />;
  }

  if (!results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white">No Results Found</h1>
          <p className="mt-4 text-lg text-gray-500 dark:text-gray-300">
            Please upload files to get matching results.
          </p>
          <button
            onClick={() => router.push("/upload")}
            className="mt-6 inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  // Calculate pagination
  const totalPages = Math.ceil(results.rankings.length / resultsPerPage);
  const indexOfLastResult = currentPage * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = results.rankings.slice(indexOfFirstResult, indexOfLastResult);

  // Function to download CV in original format
  const downloadCV = async (filename: string) => {
    try {
      // Make a request to the backend to get the original file
      const response = await fetch(`http://127.0.0.1:8000/download-file/${filename}`);
      
      if (!response.ok) {
        throw new Error('Failed to download file');
      }
      
      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Failed to download file. The original file may not be available.');
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Job Description Modal */}
      {showJobDescription && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Job Description: {results.job_description.filename}
              </h3>
              <button 
                onClick={() => setShowJobDescription(false)}
                className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="px-6 py-4 overflow-auto flex-grow">
              <pre className="text-sm text-gray-700 dark:text-gray-200 whitespace-pre-wrap">
                {results.job_description.full_text}
              </pre>
            </div>
            <div className="px-6 py-3 bg-gray-50 dark:bg-gray-700 text-right">
              <button
                onClick={() => setShowJobDescription(false)}
                className="inline-flex justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white dark:bg-gray-800 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* CV Modal */}
      {selectedCV !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                CV: {currentResults[selectedCV].filename}
              </h3>
              <button 
                onClick={() => setSelectedCV(null)}
                className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="px-6 py-4 overflow-auto flex-grow">
              <pre className="text-sm text-gray-700 dark:text-gray-200 whitespace-pre-wrap">
                {currentResults[selectedCV].full_text}
              </pre>
            </div>
            <div className="px-6 py-3 bg-gray-50 dark:bg-gray-700 flex justify-between">
              <button
                onClick={() => downloadCV(currentResults[selectedCV].filename)}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700"
              >
                Download Original
              </button>
              <button
                onClick={() => setSelectedCV(null)}
                className="inline-flex justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white dark:bg-gray-800 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="text-center mb-12">
        <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white">
          CV Matching Results
        </h1>
        <p className="mt-4 text-lg text-gray-500 dark:text-gray-300">
          We analyzed {results.total_cvs_processed} CVs against your job description
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 shadow overflow-hidden sm:rounded-lg mb-10">
        <div className="px-4 py-5 sm:px-6 flex justify-between items-center">
          <div>
            <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">
              Job Description
            </h3>
            <p className="mt-1 max-w-2xl text-sm text-gray-500 dark:text-gray-300">
              {results.job_description.filename}
            </p>
          </div>
          <button
            onClick={() => setShowJobDescription(true)}
            className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-100 dark:hover:bg-blue-800"
          >
            Show Details
          </button>
        </div>
        <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-5 sm:px-6">
          <p className="text-sm text-gray-500 dark:text-gray-300">
            {results.job_description.preview}
          </p>
        </div>
      </div>

      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Ranked CVs</h2>

      <div className="space-y-6">
        {currentResults.map((cv, index) => {
          // Debug the score
          console.log(`CV ${cv.filename}: Score ${cv.similarity_score}`);
          
          // Explicitly define thresholds
          const bestMatchThreshold = 0.8;
          const strongMatchThreshold = 0.6;
          
          // Determine match label based on score
          let matchLabel;
          let matchColor;
          
          if (cv.similarity_score >= bestMatchThreshold) {
            matchLabel = "Best Match";
            matchColor = "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100";
          } else if (cv.similarity_score >= strongMatchThreshold) {
            matchLabel = "Strong Match";
            matchColor = "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100";
          } else {
            matchLabel = "Good Match";
            matchColor = "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100";
          }
          
          return (
            <div key={index} className="bg-white dark:bg-gray-800 shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:px-6 flex justify-between items-center">
                <div>
                  <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white flex items-center">
                    {cv.filename}
                    <span className={`ml-3 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${matchColor}`}>
                      {matchLabel}
                    </span>
                  </h3>
                  <p className="mt-1 max-w-2xl text-sm text-gray-500 dark:text-gray-300">
                    Match Score: {(cv.similarity_score * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setSelectedCV(index)}
                    className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-100 dark:hover:bg-blue-800"
                  >
                    Show CV
                  </button>
                  <button
                    onClick={() => downloadCV(cv.filename)}
                    className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-green-700 bg-green-100 hover:bg-green-200 dark:bg-green-900 dark:text-green-100 dark:hover:bg-green-800"
                  >
                    Download
                  </button>
                </div>
              </div>
              <div className="border-t border-gray-200 dark:border-gray-700">
                <div className="px-4 py-5 sm:p-6">
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">
                    CV Preview
                  </h4>
                  <p className="text-sm text-gray-700 dark:text-gray-200 mb-4">
                    {cv.cv_preview}
                  </p>
                  
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">
                    Matched Keywords
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {cv.matched_keywords.map((keyword, kidx) => (
                      <span 
                        key={kidx} 
                        className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center mt-8">
          <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className={`relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white dark:bg-gray-800 text-sm font-medium ${
                currentPage === 1 
                  ? 'text-gray-300 dark:text-gray-600 cursor-not-allowed' 
                  : 'text-gray-500 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <span className="sr-only">Previous</span>
              &larr;
            </button>
            
            {[...Array(totalPages)].map((_, i) => (
              <button
                key={i}
                onClick={() => setCurrentPage(i + 1)}
                className={`relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium ${
                  currentPage === i + 1
                    ? 'z-10 bg-blue-50 dark:bg-blue-900 border-blue-500 text-blue-600 dark:text-blue-200'
                    : 'bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                {i + 1}
              </button>
            ))}
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className={`relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white dark:bg-gray-800 text-sm font-medium ${
                currentPage === totalPages 
                  ? 'text-gray-300 dark:text-gray-600 cursor-not-allowed' 
                  : 'text-gray-500 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <span className="sr-only">Next</span>
              &rarr;
            </button>
          </nav>
        </div>
      )}

      <div className="mt-10 flex justify-center">
        <button
          onClick={() => router.push("/upload")}
          className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
        >
          Start New Match
        </button>
      </div>
    </div>
  );
}