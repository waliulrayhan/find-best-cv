"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";
import { motion } from "framer-motion";

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
          <h1 className="text-3xl font-extrabold text-gray-900">No Results Found</h1>
          <p className="mt-4 text-lg text-gray-500">
            Please upload files to get matching results.
          </p>
          <button
            onClick={() => router.push("/upload")}
            className="mt-6 inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 transition-all duration-300 transform hover:scale-105"
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

  // Extract candidate name from filename
  const getCandidateName = (filename: string) => {
    // Remove file extension and replace underscores/hyphens with spaces
    return filename.replace(/\.[^/.]+$/, "").replace(/[_-]/g, " ");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-blue-50 py-12">
      {/* Job Description Modal */}
      {showJobDescription && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
          >
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-blue-50 to-indigo-50">
              <h3 className="text-lg font-medium text-gray-900">
                Job Description: {results.job_description.filename}
              </h3>
              <button 
                onClick={() => setShowJobDescription(false)}
                className="text-gray-400 hover:text-gray-500 transition-colors"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="px-6 py-4 overflow-auto flex-grow">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                {results.job_description.full_text}
              </pre>
            </div>
            <div className="px-6 py-3 bg-gray-50 text-right">
              <button
                onClick={() => setShowJobDescription(false)}
                className="inline-flex justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
              >
                Close
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* CV Modal */}
      {selectedCV !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
          >
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-blue-50 to-indigo-50">
              <h3 className="text-lg font-medium text-gray-900">
                CV: {getCandidateName(currentResults[selectedCV].filename)}
              </h3>
              <button 
                onClick={() => setSelectedCV(null)}
                className="text-gray-400 hover:text-gray-500 transition-colors"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="px-6 py-4 overflow-auto flex-grow">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                {currentResults[selectedCV].full_text}
              </pre>
            </div>
            <div className="px-6 py-3 bg-gray-50 flex justify-between">
              <button
                onClick={() => downloadCV(currentResults[selectedCV].filename)}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 transition-all duration-300 transform hover:scale-105"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Original
              </button>
              <button
                onClick={() => setSelectedCV(null)}
                className="inline-flex justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
              >
                Close
              </button>
            </div>
          </motion.div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-800">
            CV Matching Results
          </h1>
          <p className="mt-4 text-lg text-gray-600">
            We analyzed {results.total_cvs_processed} CVs against your job description
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white shadow-xl rounded-2xl overflow-hidden mb-10 border border-gray-100"
        >
          <div className="px-6 py-5 flex justify-between items-center bg-gradient-to-r from-blue-50 to-indigo-50">
            <div>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Job Description
              </h3>
              <p className="mt-1 max-w-2xl text-sm text-gray-500">
                {results.job_description.filename}
              </p>
            </div>
            <button
              onClick={() => setShowJobDescription(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-full shadow-sm text-blue-700 bg-white hover:bg-blue-50 transition-all duration-300 transform hover:scale-105"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              View Details
            </button>
          </div>
          <div className="border-t border-gray-200 px-6 py-5">
            <p className="text-sm text-gray-600">
              {results.job_description.preview}
            </p>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Ranked Candidates</h2>
            <div className="text-sm text-gray-500">
              Showing {indexOfFirstResult + 1}-{Math.min(indexOfLastResult, results.rankings.length)} of {results.rankings.length} candidates
            </div>
          </div>

          {/* Desktop Table View */}
          <div className="hidden md:block overflow-hidden shadow-lg rounded-2xl bg-white mb-8">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gradient-to-r from-blue-50 to-indigo-50">
                <tr>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-16">
                    Rank
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Candidate Name
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-32">
                    Match Score
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Matched Keywords
                  </th>
                  <th scope="col" className="px-6 py-4 text-right text-xs font-medium text-gray-500 uppercase tracking-wider w-32">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {currentResults.map((cv, index) => {
                  const rankIndex = indexOfFirstResult + index;
                  const candidateName = getCandidateName(cv.filename);
                  const matchScore = (cv.similarity_score * 100).toFixed(0);
                  
                  // Determine badge color based on score
                  let badgeColor;
                  if (cv.similarity_score >= 0.8) {
                    badgeColor = "bg-green-100 text-green-800";
                  } else if (cv.similarity_score >= 0.6) {
                    badgeColor = "bg-blue-100 text-blue-800";
                  } else {
                    badgeColor = "bg-yellow-100 text-yellow-800";
                  }
                  
                  // Determine rank medal
                  let rankMedal;
                  if (rankIndex === 0) {
                    rankMedal = "🥇";
                  } else if (rankIndex === 1) {
                    rankMedal = "🥈";
                  } else if (rankIndex === 2) {
                    rankMedal = "🥉";
                  } else {
                    rankMedal = null;
                  }
                  
                  return (
                    <motion.tr 
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                      className="hover:bg-gray-50 transition-colors duration-200"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="text-lg font-bold text-gray-900">
                          {rankMedal ? (
                            <span className="text-xl">{rankMedal}</span>
                          ) : (
                            rankIndex + 1
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{candidateName}</div>
                        <div className="text-xs text-gray-500 mt-1 truncate max-w-xs">{cv.cv_preview}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${badgeColor}`}>
                          {matchScore}%
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap gap-1">
                          {cv.matched_keywords.slice(0, 5).map((keyword, kidx) => (
                            <span 
                              key={kidx} 
                              className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-indigo-100 text-indigo-800"
                            >
                              {keyword}
                            </span>
                          ))}
                          {cv.matched_keywords.length > 5 && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-gray-800">
                              +{cv.matched_keywords.length - 5} more
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => setSelectedCV(index)}
                            className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 transition-colors duration-200"
                          >
                            View
                          </button>
                          <button
                            onClick={() => downloadCV(cv.filename)}
                            className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-green-700 bg-green-100 hover:bg-green-200 transition-colors duration-200"
                          >
                            Download
                          </button>
                        </div>
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Mobile Card View */}
          <div className="md:hidden space-y-4">
            {currentResults.map((cv, index) => {
              const rankIndex = indexOfFirstResult + index;
              const candidateName = getCandidateName(cv.filename);
              const matchScore = (cv.similarity_score * 100).toFixed(0);
              
              // Determine badge color based on score
              let badgeColor;
              if (cv.similarity_score >= 0.8) {
                badgeColor = "bg-green-100 text-green-800";
              } else if (cv.similarity_score >= 0.6) {
                badgeColor = "bg-blue-100 text-blue-800";
              } else {
                badgeColor = "bg-yellow-100 text-yellow-800";
              }
              
              // Determine rank medal
              let rankMedal;
              if (rankIndex === 0) {
                rankMedal = "🥇";
              } else if (rankIndex === 1) {
                rankMedal = "🥈";
              } else if (rankIndex === 2) {
                rankMedal = "🥉";
              } else {
                rankMedal = null;
              }
              
              return (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="bg-white rounded-xl shadow-md overflow-hidden border border-gray-100"
                >
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 flex justify-between items-center">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 mr-3">
                        <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-lg font-bold">
                          {rankMedal ? rankMedal : rankIndex + 1}
                        </div>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-gray-900">{candidateName}</h3>
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${badgeColor} mt-1`}>
                          {matchScore}% Match
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-xs text-gray-600 mb-3 line-clamp-2">{cv.cv_preview}</p>
                    
                    <h4 className="text-xs font-medium text-gray-500 mb-2">
                      Matched Keywords
                    </h4>
                    <div className="flex flex-wrap gap-1 mb-4">
                      {cv.matched_keywords.slice(0, 3).map((keyword, kidx) => (
                        <span 
                          key={kidx} 
                          className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-indigo-100 text-indigo-800"
                        >
                          {keyword}
                        </span>
                      ))}
                      {cv.matched_keywords.length > 3 && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-gray-800">
                          +{cv.matched_keywords.length - 3} more
                        </span>
                      )}
                    </div>
                    
                    <div className="flex space-x-2">
                      <button
                        onClick={() => setSelectedCV(index)}
                        className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 transition-colors duration-200"
                      >
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                        View CV
                      </button>
                      <button
                        onClick={() => downloadCV(cv.filename)}
                        className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-green-700 bg-green-100 hover:bg-green-200 transition-colors duration-200"
                      >
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center mt-8">
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className={`relative inline-flex items-center px-3 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium ${
                  currentPage === 1 
                    ? 'text-gray-300 cursor-not-allowed' 
                    : 'text-gray-500 hover:bg-gray-50'
                } transition-colors duration-200`}
              >
                <span className="sr-only">Previous</span>
                &larr;
              </button>
              
              {[...Array(totalPages)].map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentPage(i + 1)}
                  className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                    currentPage === i + 1
                      ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                      : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                  } transition-colors duration-200`}
                >
                  {i + 1}
                </button>
              ))}
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className={`relative inline-flex items-center px-3 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium ${
                  currentPage === totalPages 
                    ? 'text-gray-300 cursor-not-allowed' 
                    : 'text-gray-500 hover:bg-gray-50'
                } transition-colors duration-200`}
              >
                <span className="sr-only">Next</span>
                &rarr;
              </button>
            </nav>
          </div>
        )}

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-12 flex justify-center"
        >
          <button
            onClick={() => router.push("/upload")}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-full shadow-lg text-white bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 transform hover:scale-105 transition-all duration-300"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Start New Match
          </button>
        </motion.div>
      </div>
    </div>
  );
}