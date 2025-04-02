import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 bg-white">
      <div className="text-center">
        <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
          Find the perfect match for your job
        </h1>
        <p className="mt-5 max-w-xl mx-auto text-xl text-gray-500">
          Our AI-powered CV screening tool helps you identify the best candidates for your job openings in seconds.
        </p>
        <div className="mt-8 flex justify-center">
          <Link href="/upload" className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            Get Started
          </Link>
          <Link href="#how-it-works" className="ml-4 inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            Learn More
          </Link>
        </div>
      </div>

      <div className="mt-20" id="how-it-works">
        <h2 className="text-3xl font-extrabold text-gray-900 text-center">
          How It Works
        </h2>
        <div className="mt-12 grid gap-8 grid-cols-1 md:grid-cols-3">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                 1
              </div>
              <h3 className="mt-5 text-lg font-medium text-gray-900">Upload Job Description</h3>
              <p className="mt-2 text-base text-gray-500">
                Upload your job description document (PDF or DOCX) to define what you're looking for.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                 2
              </div>
              <h3 className="mt-5 text-lg font-medium text-gray-900">Upload CVs</h3>
              <p className="mt-2 text-base text-gray-500">
                Upload multiple candidate CVs (PDF or DOCX) that you want to evaluate.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                 3
              </div>
              <h3 className="mt-5 text-lg font-medium text-gray-900">Get Ranked Results</h3>
              <p className="mt-2 text-base text-gray-500">
                Our AI analyzes and ranks the CVs based on their relevance to your job description.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-20">
        <h2 className="text-3xl font-extrabold text-gray-900 text-center">
          Why Choose Our CV Matcher
        </h2>
        <div className="mt-12 grid gap-8 grid-cols-1 md:grid-cols-2">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900">Advanced AI Analysis</h3>
              <p className="mt-2 text-base text-gray-500">
                Our tool uses TF-IDF vectorization and cosine similarity to find the best matches based on content, not just keywords.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900">Save Time</h3>
              <p className="mt-2 text-base text-gray-500">
                Process hundreds of CVs in seconds instead of spending hours manually reviewing each one.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900">Objective Evaluation</h3>
              <p className="mt-2 text-base text-gray-500">
                Reduce unconscious bias with an algorithm that focuses purely on relevant skills and experience.
              </p>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900">Matching Keywords</h3>
              <p className="mt-2 text-base text-gray-500">
                See exactly which keywords matched between the job description and each CV.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
