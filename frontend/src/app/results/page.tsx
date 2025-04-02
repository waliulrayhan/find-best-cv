"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingState from "../components/LoadingState";

type MatchResult = {
  job_description: {
    filename: string;
    preview: string;
  };
  total_cvs_processed: number;
  rankings: {
    filename: string;
    similarity_score: number;
    cv_preview: string;
    matched_keywords: string[];
  }[];
};

export default function Results() {
  const router = useRouter();
  const [results, setResults] = useState<MatchResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterScore, setFilterScore] = useState(0);

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

  const sortedAndFilteredResults = results?.rankings
    .filter(cv => cv.similarity_score >= filterScore)
    .sort((a, b) => {
      const comparison = b.similarity_score - a.similarity_score;
      return sortOrder === 'desc' ? comparison : -comparison;
    });

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
            className="mt-6 inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  const renderControls = () => (
    <div className="mb-6 flex justify-between items-center">
      <div className="flex items-center space-x-4">
        <label className="text-sm text-gray-600">
          Minimum Score:
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={filterScore}
            onChange={(e) => setFilterScore(Number(e.target.value))}
            className="ml-2"
          />
          <span className="ml-2">{(filterScore * 100).toFixed(0)}%</span>
        </label>
      </div>
      <button
        onClick={() => setSortOrder(order => order === 'desc' ? 'asc' : 'desc')}
        className="flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50"
      >
        Sort {sortOrder === 'desc' ? '↓' : '↑'}
      </button>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-extrabold text-gray-900">
          CV Matching Results
        </h1>
        <p className="mt-4 text-lg text-gray-500">
          We analyzed {results.total_cvs_processed} CVs against your job description
        </p>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-lg mb-10">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            Job Description
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">
            {results.job_description.filename}
          </p>
        </div>
        <div className="border-t border-gray-200 px-4 py-5 sm:px-6">
          <p className="text-sm text-gray-500">
            {results.job_description.preview}
          </p>
        </div>
      </div>

      {renderControls()}

      <h2 className="text-2xl font-bold text-gray-900 mb-6">
        Ranked CV Matches
      </h2>

      <div className="space-y-6">
        {sortedAndFilteredResults?.map((cv, index) => (
          <div 
            key={index} 
            className="bg-white shadow overflow-hidden sm:rounded-lg"
          >
            <div className="px-4 py-5 sm:px-6 flex justify-between items-center">
              <div>
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  {cv.filename}
                </h3>
                <div className="mt-2 flex items-center">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-500 mr-2">
                      Match Score:
                    </span>
                    <div className="relative w-32 h-4 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="absolute top-0 left-0 h-full bg-blue-600" 
                        style={{ width: `${cv.similarity_score * 100}%` }}
                      ></div>
                    </div>
                    <span className="ml-2 text-sm font-medium text-gray-700">
                      {(cv.similarity_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  index === 0 ? 'bg-green-100 text-green-800' : 
                  index === 1 ? 'bg-blue-100 text-blue-800' : 
                  index === 2 ? 'bg-yellow-100 text-yellow-800' : 
                  'bg-gray-100 text-gray-800'
                }`}>
                  {index === 0 ? 'Best Match' : 
                   index === 1 ? 'Strong Match' : 
                   index === 2 ? 'Good Match' : 
                   'Match'}
                </span>
              </div>
            </div>
            <div className="border-t border-gray-200">
              <div className="px-4 py-5 sm:p-6">
                <h4 className="text-sm font-medium text-gray-500 mb-2">
                  CV Preview
                </h4>
                <p className="text-sm text-gray-700 mb-4">
                  {cv.cv_preview}
                </p>
                
                <h4 className="text-sm font-medium text-gray-500 mb-2">
                  Matched Keywords
                </h4>
                <div className="flex flex-wrap gap-2">
                  {cv.matched_keywords.map((keyword, kidx) => (
                    <span 
                      key={kidx} 
                      className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium bg-blue-100 text-blue-800"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10 flex justify-center">
        <button
          onClick={() => router.push("/upload")}
          className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          Start New Match
        </button>
      </div>
    </div>
  );
} 