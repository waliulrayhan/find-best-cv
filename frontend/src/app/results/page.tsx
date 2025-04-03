"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";
import { motion, AnimatePresence } from "framer-motion";
import { FiDownload, FiEye, FiPlus, FiX, FiArrowLeft, FiArrowRight, FiFileText, FiAward } from "react-icons/fi";

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
  const resultsPerPage = 10;

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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 bg-[#F8FAFC]">
        <div className="text-center">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl font-extrabold text-[#1E3A8A]">No Results Found</h1>
            <p className="mt-4 text-lg text-[#374151]">
              Please upload files to get matching results.
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => router.push("/upload")}
              className="mt-6 inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-md text-white bg-[#1E3A8A] hover:bg-[#152a61] transition-all duration-300"
            >
              <FiFileText className="mr-2 h-5 w-5" />
              Go to Upload
            </motion.button>
          </motion.div>
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
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/match-cvs', '') || 'https://cv-matcher-api.onrender.com';
      const response = await fetch(`${apiBaseUrl}/download-file/${filename}`);
      
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
    <div className="min-h-screen bg-[#F8FAFC] py-12">
      {/* Job Description Modal */}
      <AnimatePresence>
        {showJobDescription && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-gray-700 bg-opacity-40 flex items-center justify-center z-50 backdrop-blur-sm"
          >
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", damping: 25 }}
              className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
            >
              <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-[#1E3A8A]/10 to-[#10B981]/10">
                <div className="flex items-center">
                  <FiFileText className="h-5 w-5 text-[#1E3A8A] mr-2" />
                  <h3 className="text-lg font-medium text-[#1E3A8A]">
                    Job Description: {results.job_description.filename}
                  </h3>
                </div>
                <motion.button 
                  whileHover={{ rotate: 90 }}
                  transition={{ duration: 0.2 }}
                  onClick={() => setShowJobDescription(false)}
                  className="text-gray-400 hover:text-[#1E3A8A] transition-colors"
                >
                  <FiX className="h-6 w-6" />
                </motion.button>
              </div>
              <div className="px-6 py-4 overflow-auto flex-grow">
                <pre className="text-sm text-[#374151] whitespace-pre-wrap">
                  {results.job_description.full_text}
                </pre>
              </div>
              <div className="px-6 py-3 bg-gray-50 text-right">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowJobDescription(false)}
                  className="inline-flex justify-center px-4 py-2 text-sm font-medium text-[#1E3A8A] bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[#1E3A8A] transition-all duration-300"
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* CV Modal */}
      <AnimatePresence>
        {selectedCV !== null && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-gray-700 bg-opacity-40 flex items-center justify-center z-50 backdrop-blur-sm"
          >
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", damping: 25 }}
              className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col"
            >
              <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gradient-to-r from-[#1E3A8A]/10 to-[#10B981]/10">
                <div className="flex items-center">
                  <FiFileText className="h-5 w-5 text-[#1E3A8A] mr-2" />
                  <h3 className="text-lg font-medium text-[#1E3A8A]">
                    CV: {getCandidateName(currentResults[selectedCV].filename)}
                  </h3>
                </div>
                <motion.button 
                  whileHover={{ rotate: 90 }}
                  transition={{ duration: 0.2 }}
                  onClick={() => setSelectedCV(null)}
                  className="text-gray-400 hover:text-[#1E3A8A] transition-colors"
                >
                  <FiX className="h-6 w-6" />
                </motion.button>
              </div>
              <div className="px-6 py-4 overflow-auto flex-grow">
                <pre className="text-sm text-[#374151] whitespace-pre-wrap">
                  {currentResults[selectedCV].full_text}
                </pre>
              </div>
              <div className="px-6 py-3 bg-gray-50 flex justify-between">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => downloadCV(currentResults[selectedCV].filename)}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-[#10B981] hover:bg-[#0d9668] transition-all duration-300"
                >
                  <FiDownload className="w-4 h-4 mr-2" />
                  Download Original
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setSelectedCV(null)}
                  className="inline-flex justify-center px-4 py-2 text-sm font-medium text-[#374151] bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[#1E3A8A] transition-all duration-300"
                >
                  Close
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-[#1E3A8A] to-[#10B981]">
            CV Matching Results
          </h1>
          <p className="mt-4 text-lg text-[#374151]">
            We analyzed {results.total_cvs_processed} CVs against your job description
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white shadow-xl rounded-2xl overflow-hidden mb-10 border border-gray-100"
        >
          <div className="px-6 py-5 flex justify-between items-center bg-gradient-to-r from-[#1E3A8A]/10 to-[#10B981]/10">
            <div className="flex items-center">
              <motion.div
                animate={{ rotate: [0, 10, -10, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 5 }}
              >
                <FiFileText className="h-6 w-6 text-[#1E3A8A] mr-3" />
              </motion.div>
              <div>
                <h3 className="text-lg leading-6 font-medium text-[#1E3A8A]">
                  Job Description
                </h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">
                  {results.job_description.filename}
                </p>
              </div>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowJobDescription(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-full shadow-sm text-[#1E3A8A] bg-white hover:bg-[#1E3A8A]/10 transition-all duration-300"
            >
              <FiEye className="w-4 h-4 mr-2" />
              View Details
            </motion.button>
          </div>
          <div className="border-t border-gray-200 px-6 py-5">
            <p className="text-sm text-[#374151]">
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
            <div className="flex items-center">
              <motion.div
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 2 }}
              >
                <FiAward className="h-6 w-6 text-[#F59E0B] mr-2" />
              </motion.div>
              <h2 className="text-2xl font-bold text-[#1E3A8A]">Ranked Candidates</h2>
            </div>
            <div className="text-sm text-[#374151]">
              Showing {indexOfFirstResult + 1}-{Math.min(indexOfLastResult, results.rankings.length)} of {results.rankings.length} candidates
            </div>
          </div>

          {/* Desktop Table View */}
          <div className="hidden md:block overflow-hidden shadow-lg rounded-2xl bg-white mb-8">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gradient-to-r from-[#1E3A8A]/10 to-[#10B981]/10">
                <tr>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-[#374151] uppercase tracking-wider w-16">
                    Rank
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-[#374151] uppercase tracking-wider">
                    Candidate Name
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-[#374151] uppercase tracking-wider w-32">
                    Match Score
                  </th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-medium text-[#374151] uppercase tracking-wider">
                    Matched Keywords
                  </th>
                  <th scope="col" className="px-6 py-4 text-right text-xs font-medium text-[#374151] uppercase tracking-wider w-32">
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
                    badgeColor = "bg-[#10B981]/20 text-[#10B981]";
                  } else if (cv.similarity_score >= 0.6) {
                    badgeColor = "bg-[#1E3A8A]/20 text-[#1E3A8A]";
                  } else {
                    badgeColor = "bg-[#F59E0B]/20 text-[#F59E0B]";
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
                      className="hover:bg-[#F8FAFC] transition-colors duration-200"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="text-lg font-bold text-[#374151]">
                          {rankMedal ? (
                            <motion.span 
                              className="text-xl"
                              animate={{ scale: [1, 1.2, 1] }}
                              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                            >
                              {rankMedal}
                            </motion.span>
                          ) : (
                            rankIndex + 1
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-[#1E3A8A]">{candidateName}</div>
                        <div className="text-xs text-[#374151] mt-1 truncate max-w-xs">{cv.cv_preview}</div>
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
                              className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-[#1E3A8A]/10 text-[#1E3A8A]"
                            >
                              {keyword}
                            </span>
                          ))}
                          {cv.matched_keywords.length > 5 && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-[#374151]">
                              +{cv.matched_keywords.length - 5} more
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end space-x-2">
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setSelectedCV(index)}
                            className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-[#1E3A8A] bg-[#1E3A8A]/10 hover:bg-[#1E3A8A]/20 transition-colors duration-200"
                          >
                            <FiEye className="w-3.5 h-3.5 mr-1" />
                            View
                          </motion.button>
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => downloadCV(cv.filename)}
                            className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-[#10B981] bg-[#10B981]/10 hover:bg-[#10B981]/20 transition-colors duration-200"
                          >
                            <FiDownload className="w-3.5 h-3.5 mr-1" />
                            Download
                          </motion.button>
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
                badgeColor = "bg-[#10B981]/20 text-[#10B981]";
              } else if (cv.similarity_score >= 0.6) {
                badgeColor = "bg-[#1E3A8A]/20 text-[#1E3A8A]";
              } else {
                badgeColor = "bg-[#F59E0B]/20 text-[#F59E0B]";
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
                  <div className="p-4 bg-gradient-to-r from-[#1E3A8A]/10 to-[#10B981]/10 flex justify-between items-center">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 mr-3">
                        <motion.div 
                          className="w-10 h-10 rounded-full bg-[#1E3A8A]/10 flex items-center justify-center text-lg font-bold"
                          whileHover={{ scale: 1.1, rotate: 5 }}
                        >
                          {rankMedal ? (
                            <motion.span
                              animate={{ scale: [1, 1.2, 1] }}
                              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                            >
                              {rankMedal}
                            </motion.span>
                          ) : (
                            rankIndex + 1
                          )}
                        </motion.div>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-[#1E3A8A]">{candidateName}</h3>
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${badgeColor} mt-1`}>
                          {matchScore}% Match
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-xs text-[#374151] mb-3 line-clamp-2">{cv.cv_preview}</p>
                    
                    <h4 className="text-xs font-medium text-[#374151] mb-2">
                      Matched Keywords
                    </h4>
                    <div className="flex flex-wrap gap-1 mb-4">
                      {cv.matched_keywords.slice(0, 3).map((keyword, kidx) => (
                        <span 
                          key={kidx} 
                          className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-[#1E3A8A]/10 text-[#1E3A8A]"
                        >
                          {keyword}
                        </span>
                      ))}
                      {cv.matched_keywords.length > 3 && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-[#374151]">
                          +{cv.matched_keywords.length - 3} more
                        </span>
                      )}
                    </div>
                    
                    <div className="flex space-x-2">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setSelectedCV(index)}
                        className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-[#1E3A8A] bg-[#1E3A8A]/10 hover:bg-[#1E3A8A]/20 transition-colors duration-200"
                      >
                        <FiEye className="w-4 h-4 mr-1" />
                        View CV
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => downloadCV(cv.filename)}
                        className="flex-1 inline-flex justify-center items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-[#10B981] bg-[#10B981]/10 hover:bg-[#10B981]/20 transition-colors duration-200"
                      >
                        <FiDownload className="w-4 h-4 mr-1" />
                        Download
                      </motion.button>
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
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className={`relative inline-flex items-center px-3 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium ${
                  currentPage === 1 
                    ? 'text-gray-300 cursor-not-allowed' 
                    : 'text-[#1E3A8A] hover:bg-[#1E3A8A]/10'
                } transition-colors duration-200`}
              >
                <span className="sr-only">Previous</span>
                <FiArrowLeft className="h-4 w-4" />
              </motion.button>
              
              {[...Array(totalPages)].map((_, i) => (
                <motion.button
                  key={i}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setCurrentPage(i + 1)}
                  className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                    currentPage === i + 1
                      ? 'z-10 bg-[#1E3A8A]/10 border-[#1E3A8A] text-[#1E3A8A]'
                      : 'bg-white border-gray-300 text-[#374151] hover:bg-[#F8FAFC]'
                  } transition-colors duration-200`}
                >
                  {i + 1}
                </motion.button>
              ))}
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className={`relative inline-flex items-center px-3 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium ${
                  currentPage === totalPages 
                    ? 'text-gray-300 cursor-not-allowed' 
                    : 'text-[#1E3A8A] hover:bg-[#1E3A8A]/10'
                } transition-colors duration-200`}
              >
                <span className="sr-only">Next</span>
                <FiArrowRight className="h-4 w-4" />
              </motion.button>
            </nav>
          </div>
        )}

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-12 flex justify-center"
        >
          <motion.button
            whileHover={{ scale: 1.05, boxShadow: "0 10px 25px -5px rgba(30, 58, 138, 0.3)" }}
            whileTap={{ scale: 0.97 }}
            onClick={() => router.push("/upload")}
            className="inline-flex items-center px-8 py-4 border border-transparent text-base font-medium rounded-xl shadow-xl text-white bg-gradient-to-r from-[#1E3A8A] to-[#10B981] hover:from-[#152a61] hover:to-[#0d9668] transition-all duration-300"
          >
            <motion.div
              animate={{ rotate: [0, 180, 360] }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear", repeatDelay: 10 }}
            >
              <FiPlus className="w-5 h-5 mr-3" />
            </motion.div>
            Start New Match
          </motion.button>
        </motion.div>
      </div>
    </div>
  );
}