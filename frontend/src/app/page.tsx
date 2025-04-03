import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 bg-[#F8FAFC]">
      {/* Hero Section */}
      <div className="text-center py-16">
        <h1 className="text-4xl font-extrabold text-[#1E3A8A] sm:text-5xl sm:tracking-tight lg:text-6xl">
          AI-Powered CV Screening for Smart Hiring
        </h1>
        <p className="mt-5 max-w-xl mx-auto text-xl text-[#374151]">
          Upload resumes and let AI rank the best candidates for you.
        </p>
        <div className="mt-8 flex justify-center">
          <a 
            href="#how-it-works" 
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-[#1E3A8A] hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-300"
          >
            Start Now
          </a>
          <Link 
            href="/upload" 
            className="ml-4 inline-flex items-center px-6 py-3 border border-[#10B981] shadow-sm text-base font-medium rounded-md text-[#10B981] bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#10B981] transition-all duration-300"
          >
            Upload CVs
          </Link>
        </div>
        <div className="mt-10 flex justify-center">
          <Image 
            src="/images/ai-resume-analysis.svg" 
            alt="AI Resume Analysis" 
            width={500} 
            height={300}
            className="animate-fade-in"
          />
        </div>
      </div>

      {/* Features Section */}
      <div className="mt-20 animate-fade-in">
        <h2 className="text-3xl font-extrabold text-[#1E3A8A] text-center">
          Why Choose Our Platform
        </h2>
        <div className="mt-12 grid gap-8 grid-cols-1 md:grid-cols-3">
          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#1E3A8A] text-white mb-4">
                📄
              </div>
              <h3 className="text-lg font-medium text-[#374151]">Intelligent Screening</h3>
              <p className="mt-2 text-base text-gray-500">
                Our AI ranks CVs based on job relevance, finding the perfect match for your requirements.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#10B981] text-white mb-4">
                📊
              </div>
              <h3 className="text-lg font-medium text-[#374151]">Data-Driven Insights</h3>
              <p className="mt-2 text-base text-gray-500">
                Find top candidates with AI-powered analysis that goes beyond simple keyword matching.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#F59E0B] text-white mb-4">
                ⚡
              </div>
              <h3 className="text-lg font-medium text-[#374151]">Fast & Efficient</h3>
              <p className="mt-2 text-base text-gray-500">
                Save time and hire smarter with our lightning-fast CV processing and ranking system.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="mt-20 animate-fade-in" id="how-it-works">
        <h2 className="text-3xl font-extrabold text-[#1E3A8A] text-center">
          How It Works
        </h2>
        <div className="mt-12 grid gap-8 grid-cols-1 md:grid-cols-3">
          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#1E3A8A] text-white">
                1
              </div>
              <h3 className="mt-5 text-lg font-medium text-[#374151]">Upload CVs</h3>
              <p className="mt-2 text-base text-gray-500">
                Upload multiple candidate CVs (PDF or DOCX) that you want to evaluate.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#10B981] text-white">
                2
              </div>
              <h3 className="mt-5 text-lg font-medium text-[#374151]">Enter Job Description</h3>
              <p className="mt-2 text-base text-gray-500">
                Upload or enter your job description to define what you're looking for in candidates.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-[#F59E0B] text-white">
                3
              </div>
              <h3 className="mt-5 text-lg font-medium text-[#374151]">Get Ranked Candidates</h3>
              <p className="mt-2 text-base text-gray-500">
                Our AI analyzes and ranks the CVs based on their relevance to your job description.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="mt-20 animate-fade-in">
        <h2 className="text-3xl font-extrabold text-[#1E3A8A] text-center">
          What Recruiters Say
        </h2>
        <div className="mt-12 grid gap-8 grid-cols-1 md:grid-cols-2">
          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center mb-4">
                <div className="h-12 w-12 rounded-full overflow-hidden mr-4">
                  <Image src="/images/testimonial-1.jpg" alt="Sarah Johnson" width={48} height={48} className="h-full w-full object-cover" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#374151]">Sarah Johnson</h3>
                  <p className="text-sm text-gray-500">HR Director, TechCorp</p>
                </div>
              </div>
              <div className="flex text-[#F59E0B] mb-2">
                ★★★★★
              </div>
              <p className="text-base text-gray-500">
                "This tool has revolutionized our hiring process. We've reduced our screening time by 70% and found better candidates."
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow-lg rounded-lg hover:shadow-xl transition-shadow duration-300">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center mb-4">
                <div className="h-12 w-12 rounded-full overflow-hidden mr-4">
                  <Image src="/images/testimonial-2.jpg" alt="Mark Thompson" width={48} height={48} className="h-full w-full object-cover" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#374151]">Mark Thompson</h3>
                  <p className="text-sm text-gray-500">Talent Acquisition, FinanceHub</p>
                </div>
              </div>
              <div className="flex text-[#F59E0B] mb-2">
                ★★★★★
              </div>
              <p className="text-base text-gray-500">
                "The AI matching is incredibly accurate. We've made better hiring decisions and saved countless hours in the process."
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
