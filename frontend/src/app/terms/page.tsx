import React from 'react';
import Link from 'next/link';

export default function TermsPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="bg-white shadow-sm rounded-lg p-6 sm:p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Terms of Service</h1>
        
        <div className="prose prose-blue max-w-none">
          <p className="text-gray-600 mb-4">
            Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">1. Introduction</h2>
          <p className="text-gray-600 mb-4">
            Welcome to CV Matcher. These terms and conditions outline the rules and regulations for the use of our website and services.
          </p>
          <p className="text-gray-600 mb-4">
            By accessing this website, we assume you accept these terms and conditions in full. Do not continue to use CV Matcher if you do not accept all of the terms and conditions stated on this page.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">2. License to Use</h2>
          <p className="text-gray-600 mb-4">
            Unless otherwise stated, CV Matcher and/or its licensors own the intellectual property rights for all material on CV Matcher. All intellectual property rights are reserved.
          </p>
          <p className="text-gray-600 mb-4">
            You may view and/or print pages from the website for your own personal use subject to restrictions set in these terms and conditions.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">3. Restrictions</h2>
          <p className="text-gray-600 mb-4">
            You are specifically restricted from all of the following:
          </p>
          <ul className="list-disc pl-5 text-gray-600 mb-4">
            <li className="mb-2">Publishing any website material in any other media.</li>
            <li className="mb-2">Selling, sublicensing and/or otherwise commercializing any website material.</li>
            <li className="mb-2">Using this website in any way that is or may be damaging to this website.</li>
            <li className="mb-2">Using this website in any way that impacts user access to this website.</li>
            <li className="mb-2">Using this website contrary to applicable laws and regulations, or in any way may cause harm to the website, or to any person or business entity.</li>
          </ul>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">4. Your Content</h2>
          <p className="text-gray-600 mb-4">
            In these terms and conditions, "Your Content" shall mean any audio, video, text, images or other material you choose to display on this website. By displaying Your Content, you grant CV Matcher a non-exclusive, worldwide, irrevocable, royalty-free, sublicensable license to use, reproduce, adapt, publish, translate and distribute it in any and all media.
          </p>
          <p className="text-gray-600 mb-4">
            Your Content must be your own and must not be infringing on any third party's rights. CV Matcher reserves the right to remove any of Your Content from this website at any time without notice.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">5. No Warranties</h2>
          <p className="text-gray-600 mb-4">
            This website is provided "as is," with all faults, and CV Matcher makes no express or implied representations or warranties, of any kind related to this website or the materials contained on this website.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">6. Limitation of Liability</h2>
          <p className="text-gray-600 mb-4">
            In no event shall CV Matcher, nor any of its officers, directors and employees, be held liable for anything arising out of or in any way connected with your use of this website, whether such liability is under contract, tort or otherwise.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">7. Governing Law & Jurisdiction</h2>
          <p className="text-gray-600 mb-4">
            These terms will be governed by and interpreted in accordance with the laws of the country/state where CV Matcher is based, and you submit to the non-exclusive jurisdiction of the state and federal courts located there for the resolution of any disputes.
          </p>
        </div>
        
        <div className="mt-8 pt-6 border-t border-gray-200">
          <Link 
            href="/"
            className="inline-flex items-center text-[#1E3A8A] hover:text-[#10B981] transition-colors duration-300"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
} 