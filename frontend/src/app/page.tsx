import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-blue-50">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-24">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-800 tracking-tight">
            AI-Powered CV Screening for Smart Hiring
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-600 leading-relaxed">
            Upload resumes and let our intelligent AI find your perfect candidates in seconds.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <a 
              href="#how-it-works" 
              className="px-8 py-4 text-base font-medium rounded-full bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              See How It Works
            </a>
            <Link 
              href="/upload" 
              className="px-8 py-4 text-base font-medium rounded-full bg-white text-blue-600 border border-blue-200 shadow-md hover:shadow-lg hover:border-blue-300 transform hover:-translate-y-1 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Upload CVs
            </Link>
          </div>
        </div>
        <div className="mt-16 flex justify-center">
          <div className="relative w-full max-w-4xl">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur opacity-20"></div>
            <div className="relative overflow-hidden rounded-2xl shadow-2xl">
              <Image 
                src="/images/ai-resume-analysis.svg" 
                alt="AI Resume Analysis" 
                width={1000} 
                height={600}
                className="w-full h-auto animate-fade-in"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
            Why Choose Our Platform
          </h2>
          <div className="mt-4 h-1 w-24 bg-blue-600 mx-auto rounded-full"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-3">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mb-6">
                <span className="text-3xl">📄</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Intelligent Screening</h3>
              <p className="text-gray-600">
                Our AI ranks CVs based on job relevance, finding the perfect match for your requirements.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-6">
                <span className="text-3xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Data-Driven Insights</h3>
              <p className="text-gray-600">
                Find top candidates with AI-powered analysis that goes beyond simple keyword matching.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl overflow-hidden transform hover:scale-105 transition-all duration-300">
            <div className="p-8">
              <div className="w-16 h-16 rounded-full bg-amber-100 flex items-center justify-center mb-6">
                <span className="text-3xl">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Fast & Efficient</h3>
              <p className="text-gray-600">
                Save time and hire smarter with our lightning-fast CV processing and ranking system.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 bg-white" id="how-it-works">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
            How It Works
          </h2>
          <div className="mt-4 h-1 w-24 bg-blue-600 mx-auto rounded-full"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-3">
          <div className="relative">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-400 to-blue-600 rounded-2xl blur opacity-20"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-blue-50">
              <div className="w-14 h-14 rounded-full bg-blue-600 text-white flex items-center justify-center text-2xl font-bold mb-6">
                1
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Upload CVs</h3>
              <p className="text-gray-600">
                Upload multiple candidate CVs (PDF or DOCX) that you want to evaluate.
              </p>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-1 bg-gradient-to-r from-green-400 to-green-600 rounded-2xl blur opacity-20"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-green-50">
              <div className="w-14 h-14 rounded-full bg-green-600 text-white flex items-center justify-center text-2xl font-bold mb-6">
                2
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Enter Job Description</h3>
              <p className="text-gray-600">
                Upload or enter your job description to define what you're looking for in candidates.
              </p>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-1 bg-gradient-to-r from-amber-400 to-amber-600 rounded-2xl blur opacity-20"></div>
            <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-amber-50">
              <div className="w-14 h-14 rounded-full bg-amber-500 text-white flex items-center justify-center text-2xl font-bold mb-6">
                3
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Get Ranked Candidates</h3>
              <p className="text-gray-600">
                Our AI analyzes and ranks the CVs based on their relevance to your job description.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
            What Recruiters Say
          </h2>
          <div className="mt-4 h-1 w-24 bg-blue-600 mx-auto rounded-full"></div>
        </div>
        <div className="grid gap-10 grid-cols-1 md:grid-cols-2">
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <div className="flex flex-col sm:flex-row items-start sm:items-center mb-6">
              <div className="h-16 w-16 rounded-full overflow-hidden mr-4 ring-4 ring-blue-100">
                <Image src="/images/testimonial-1.jpg" alt="Sarah Johnson" width={64} height={64} className="h-full w-full object-cover" />
              </div>
              <div className="mt-4 sm:mt-0">
                <h3 className="text-xl font-semibold text-gray-900">Sarah Johnson</h3>
                <p className="text-blue-600">HR Director, TechCorp</p>
              </div>
            </div>
            <div className="flex text-amber-400 mb-4">
              <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
            </div>
            <p className="text-gray-600 italic">
              "This tool has revolutionized our hiring process. We've reduced our screening time by 70% and found better candidates."
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <div className="flex flex-col sm:flex-row items-start sm:items-center mb-6">
              <div className="h-16 w-16 rounded-full overflow-hidden mr-4 ring-4 ring-blue-100">
                <Image src="/images/testimonial-2.jpg" alt="Mark Thompson" width={64} height={64} className="h-full w-full object-cover" />
              </div>
              <div className="mt-4 sm:mt-0">
                <h3 className="text-xl font-semibold text-gray-900">Mark Thompson</h3>
                <p className="text-blue-600">Talent Acquisition, FinanceHub</p>
              </div>
            </div>
            <div className="flex text-amber-400 mb-4">
              <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
            </div>
            <p className="text-gray-600 italic">
              "The AI matching is incredibly accurate. We've made better hiring decisions and saved countless hours in the process."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
