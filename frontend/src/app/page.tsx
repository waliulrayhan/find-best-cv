import Image from "next/image";
import Link from "next/link";
import { FaRocket, FaChartBar, FaBolt, FaFileUpload, FaFileAlt, FaUserCheck } from "react-icons/fa";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#F8FAFC] to-[#E2E8F0]">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-28">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-[#1E3A8A] to-[#10B981] tracking-tight animate-fadeIn">
            CV Screening for Smart Hiring
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-[#374151] leading-relaxed animate-slideUp">
            Upload resumes and find your perfect candidates in seconds.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4 animate-fadeIn">
            <a 
              href="#how-it-works" 
              className="group px-8 py-4 text-base font-medium rounded-full bg-gradient-to-r from-[#1E3A8A] to-[#10B981] text-white shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#1E3A8A]"
            >
              <span className="flex items-center gap-2">
                <FaRocket className="inline-block transform group-hover:rotate-12 transition-transform duration-300 animate-bounce" />
                See How It Works
              </span>
            </a>
            <Link 
              href="/upload" 
              className="group px-8 py-4 text-base font-medium rounded-full bg-white text-[#1E3A8A] border border-[#1E3A8A]/20 shadow-md hover:shadow-lg hover:border-[#1E3A8A]/40 transform hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#1E3A8A]"
            >
              <span className="flex items-center gap-2">
                <FaFileUpload className="inline-block transform group-hover:scale-110 transition-transform duration-300 animate-pulse" />
                Upload CVs
              </span>
            </Link>
          </div>
        </div>
        <div className="mt-16 flex justify-center animate-fadeIn">
          <div className="relative w-full max-w-4xl">
            <div className="absolute -inset-1 bg-gradient-to-r from-[#1E3A8A] to-[#10B981] rounded-2xl blur opacity-20 animate-pulse"></div>
            <div className="relative overflow-hidden rounded-2xl shadow-2xl">
              <Image 
                src="/bg.jpg" 
                alt="AI Resume Analysis" 
                width={1000} 
                height={600}
                className="w-full h-auto animate-float"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <div className="text-center mb-16 animate-fadeIn">
          <h2 className="text-3xl md:text-4xl font-bold text-[#374151]">
            Why Choose Our Platform
          </h2>
          <div className="mt-4 h-1 w-24 bg-[#1E3A8A] mx-auto rounded-full animate-expand"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-3">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300 border border-[#1E3A8A]/5 hover:border-[#1E3A8A]/20 animate-slideUp">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-[#1E3A8A]/10 flex items-center justify-center mb-6 transform transition-all duration-500 hover:rotate-12 animate-pulse">
                <FaFileAlt className="text-3xl text-[#1E3A8A]" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Intelligent Screening</h3>
              <p className="text-[#374151]/80">
                Our AI ranks CVs based on job relevance, finding the perfect match for your requirements.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300 border border-[#10B981]/5 hover:border-[#10B981]/20 animate-slideUp animation-delay-200">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-[#10B981]/10 flex items-center justify-center mb-6 transform transition-all duration-500 hover:rotate-12 animate-pulse">
                <FaChartBar className="text-3xl text-[#10B981]" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Data-Driven Insights</h3>
              <p className="text-[#374151]/80">
                Find top candidates with AI-powered analysis that goes beyond simple keyword matching.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300 border border-[#F59E0B]/5 hover:border-[#F59E0B]/20 animate-slideUp animation-delay-400">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-[#F59E0B]/10 flex items-center justify-center mb-6 transform transition-all duration-500 hover:rotate-12 animate-pulse">
                <FaBolt className="text-3xl text-[#F59E0B]" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Fast & Efficient</h3>
              <p className="text-[#374151]/80">
                Save time and hire smarter with our lightning-fast CV processing and ranking system.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 bg-white" id="how-it-works">
        <div className="text-center mb-16 animate-fadeIn">
          <h2 className="text-3xl md:text-4xl font-bold text-[#374151]">
            How It Works
          </h2>
          <div className="mt-4 h-1 w-24 bg-[#1E3A8A] mx-auto rounded-full animate-expand"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-3">
          <div className="relative group animate-slideRight">
            <div className="absolute -inset-1 bg-gradient-to-r from-[#1E3A8A] to-[#3B82F6] rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity duration-300 animate-pulse"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-[#1E3A8A]/10 group-hover:border-[#1E3A8A]/20 transition-all duration-300">
              <div className="w-14 h-14 rounded-full bg-[#1E3A8A] text-white flex items-center justify-center text-2xl font-bold mb-6 group-hover:scale-110 transition-transform duration-300">
                <FaFileUpload className="text-xl animate-bounce" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Upload CVs</h3>
              <p className="text-[#374151]/80">
                Upload multiple candidate CVs (PDF or DOCX) that you want to evaluate.
              </p>
            </div>
          </div>

          <div className="relative group animate-slideRight animation-delay-200">
            <div className="absolute -inset-1 bg-gradient-to-r from-[#10B981] to-[#34D399] rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity duration-300 animate-pulse"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-[#10B981]/10 group-hover:border-[#10B981]/20 transition-all duration-300">
              <div className="w-14 h-14 rounded-full bg-[#10B981] text-white flex items-center justify-center text-2xl font-bold mb-6 group-hover:scale-110 transition-transform duration-300">
                <FaFileAlt className="text-xl animate-pulse" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Enter Job Description</h3>
              <p className="text-[#374151]/80">
                Upload or enter your job description to define what you're looking for in candidates.
              </p>
            </div>
          </div>

          <div className="relative group animate-slideRight animation-delay-400">
            <div className="absolute -inset-1 bg-gradient-to-r from-[#F59E0B] to-[#FBBF24] rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity duration-300 animate-pulse"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-[#F59E0B]/10 group-hover:border-[#F59E0B]/20 transition-all duration-300">
              <div className="w-14 h-14 rounded-full bg-[#F59E0B] text-white flex items-center justify-center text-2xl font-bold mb-6 group-hover:scale-110 transition-transform duration-300">
                <FaUserCheck className="text-xl animate-spin-slow" />
              </div>
              <h3 className="text-xl font-semibold text-[#374151] mb-3">Get Ranked Candidates</h3>
              <p className="text-[#374151]/80">
                Our AI analyzes and ranks the CVs based on their relevance to your job description.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <div className="text-center mb-16 animate-fadeIn">
          <h2 className="text-3xl md:text-4xl font-bold text-[#374151]">
            What Recruiters Say
          </h2>
          <div className="mt-4 h-1 w-24 bg-[#1E3A8A] mx-auto rounded-full animate-expand"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-2">
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-[#1E3A8A]/5 hover:border-[#1E3A8A]/20 transition-all duration-300 transform hover:scale-[1.02] animate-fadeIn">
            <div className="flex flex-col sm:flex-row items-start sm:items-center mb-6">
              <div className="h-16 w-16 rounded-full overflow-hidden mr-4 ring-4 ring-[#1E3A8A]/20 animate-pulse">
                <Image src="/youngman.png" alt="Sarah Johnson" width={64} height={64} className="h-full w-full object-cover" />
              </div>
              <div className="mt-4 sm:mt-0">
                <h3 className="text-xl font-semibold text-[#374151]">Sarah Johnson</h3>
                <p className="text-[#1E3A8A]">HR Director, TechCorp</p>
              </div>
            </div>
            <div className="flex text-[#F59E0B] mb-4 animate-twinkle">
              <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
            </div>
            <p className="text-[#374151]/80 italic">
              "This tool has revolutionized our hiring process. We've reduced our screening time by 70% and found better candidates."
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-8 border border-[#1E3A8A]/5 hover:border-[#1E3A8A]/20 transition-all duration-300 transform hover:scale-[1.02] animate-fadeIn animation-delay-200">
            <div className="flex flex-col sm:flex-row items-start sm:items-center mb-6">
              <div className="h-16 w-16 rounded-full overflow-hidden mr-4 ring-4 ring-[#1E3A8A]/20 animate-pulse">
                <Image src="/bussinessman.png" alt="Mark Thompson" width={64} height={64} className="h-full w-full object-cover" />
              </div>
              <div className="mt-4 sm:mt-0">
                <h3 className="text-xl font-semibold text-[#374151]">Mark Thompson</h3>
                <p className="text-[#1E3A8A]">Talent Acquisition, FinanceHub</p>
              </div>
            </div>
            <div className="flex text-[#F59E0B] mb-4 animate-twinkle">
              <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
            </div>
            <p className="text-[#374151]/80 italic">
              "The AI matching is incredibly accurate. We've made better hiring decisions and saved countless hours in the process."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
