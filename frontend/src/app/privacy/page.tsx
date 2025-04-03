import React from 'react';
import Link from 'next/link';

export default function PrivacyPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="bg-white shadow-sm rounded-lg p-6 sm:p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Privacy Policy</h1>
        
        <div className="prose prose-blue max-w-none">
          <p className="text-gray-600 mb-4">
            Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">1. Introduction</h2>
          <p className="text-gray-600 mb-4">
            Welcome to CV Matcher. We respect your privacy and are committed to protecting your personal data. 
            This privacy policy will inform you about how we look after your personal data when you visit our website 
            and tell you about your privacy rights and how the law protects you.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">2. Data We Collect</h2>
          <p className="text-gray-600 mb-4">
            We may collect, use, store and transfer different kinds of personal data about you which we have grouped together as follows:
          </p>
          <ul className="list-disc pl-5 text-gray-600 mb-4">
            <li className="mb-2">Identity Data includes first name, last name, username or similar identifier.</li>
            <li className="mb-2">Contact Data includes email address and telephone numbers.</li>
            <li className="mb-2">Technical Data includes internet protocol (IP) address, browser type and version, time zone setting and location, browser plug-in types and versions, operating system and platform, and other technology on the devices you use to access this website.</li>
            <li className="mb-2">Usage Data includes information about how you use our website and services.</li>
            <li className="mb-2">CV Data includes information you provide in your curriculum vitae for job matching purposes.</li>
          </ul>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">3. How We Use Your Data</h2>
          <p className="text-gray-600 mb-4">
            We will only use your personal data when the law allows us to. Most commonly, we will use your personal data in the following circumstances:
          </p>
          <ul className="list-disc pl-5 text-gray-600 mb-4">
            <li className="mb-2">To provide our CV matching services.</li>
            <li className="mb-2">To improve our website and services.</li>
            <li className="mb-2">To communicate with you about our services.</li>
            <li className="mb-2">To comply with legal obligations.</li>
          </ul>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">4. Data Security</h2>
          <p className="text-gray-600 mb-4">
            We have put in place appropriate security measures to prevent your personal data from being accidentally lost, 
            used or accessed in an unauthorized way, altered or disclosed. In addition, we limit access to your personal data 
            to those employees, agents, contractors and other third parties who have a business need to know.
          </p>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">5. Your Legal Rights</h2>
          <p className="text-gray-600 mb-4">
            Under certain circumstances, you have rights under data protection laws in relation to your personal data, including:
          </p>
          <ul className="list-disc pl-5 text-gray-600 mb-4">
            <li className="mb-2">The right to access your personal data.</li>
            <li className="mb-2">The right to correction of your personal data.</li>
            <li className="mb-2">The right to erasure of your personal data.</li>
            <li className="mb-2">The right to object to processing of your personal data.</li>
            <li className="mb-2">The right to data portability.</li>
          </ul>
          
          <h2 className="text-xl font-semibold text-gray-800 mt-8 mb-4">6. Contact Us</h2>
          <p className="text-gray-600 mb-4">
            If you have any questions about this privacy policy or our privacy practices, please contact us at:
          </p>
          <p className="text-gray-600 mb-4">
            Email: privacy@cvmatcher.com<br />
            Phone: +1 (555) 123-4567
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