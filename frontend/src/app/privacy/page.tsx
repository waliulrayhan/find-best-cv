"use client"

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-[#F8FAFC] py-12">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8"
      >
        <div className="bg-white shadow-lg rounded-xl p-6 sm:p-8 border-l-4 border-[#1E3A8A]">
          <div className="flex items-center mb-6">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="mr-4 text-[#1E3A8A]"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </motion.div>
            <h1 className="text-3xl font-bold text-[#374151] flex-1">Privacy Policy</h1>
          </div>
          
          <div className="prose prose-blue max-w-none">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="bg-[#F8FAFC] p-4 rounded-lg mb-6 border-l-2 border-[#F59E0B]"
            >
              <p className="text-[#374151] font-medium">
                Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">1</span>
                Introduction
              </h2>
              <p className="text-[#374151] mb-4">
                Welcome to CV Matcher. We respect your privacy and are committed to protecting your personal data. 
                This privacy policy will inform you about how we look after your personal data when you visit our website 
                and tell you about your privacy rights and how the law protects you.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">2</span>
                Data We Collect
              </h2>
              <p className="text-[#374151] mb-4">
                We may collect, use, store and transfer different kinds of personal data about you which we have grouped together as follows:
              </p>
              <ul className="list-none pl-5 text-[#374151] mb-4 space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Identity Data includes first name, last name, username or similar identifier.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Contact Data includes email address and telephone numbers.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Technical Data includes internet protocol (IP) address, browser type and version, time zone setting and location, browser plug-in types and versions, operating system and platform, and other technology on the devices you use to access this website.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Usage Data includes information about how you use our website and services.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  CV Data includes information you provide in your curriculum vitae for job matching purposes.
                </li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">3</span>
                How We Use Your Data
              </h2>
              <p className="text-[#374151] mb-4">
                We will only use your personal data when the law allows us to. Most commonly, we will use your personal data in the following circumstances:
              </p>
              <ul className="list-none pl-5 text-[#374151] mb-4 space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  To provide our CV matching services.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  To improve our website and services.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  To communicate with you about our services.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#10B981] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  To comply with legal obligations.
                </li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">4</span>
                Data Security
              </h2>
              <p className="text-[#374151] mb-4">
                We have put in place appropriate security measures to prevent your personal data from being accidentally lost, 
                used or accessed in an unauthorized way, altered or disclosed. In addition, we limit access to your personal data 
                to those employees, agents, contractors and other third parties who have a business need to know.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">5</span>
                Your Legal Rights
              </h2>
              <p className="text-[#374151] mb-4">
                Under certain circumstances, you have rights under data protection laws in relation to your personal data, including:
              </p>
              <ul className="list-none pl-5 text-[#374151] mb-4 space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  The right to access your personal data.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  The right to correction of your personal data.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  The right to erasure of your personal data.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  The right to object to processing of your personal data.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  The right to data portability.
                </li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">6</span>
                Contact Us
              </h2>
              <div className="bg-[#F8FAFC] p-5 rounded-lg border border-[#1E3A8A] shadow-sm">
                <p className="text-[#374151] mb-4">
                  If you have any questions about this privacy policy or our privacy practices, please contact us at:
                </p>
                <p className="text-[#374151] flex items-center mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#1E3A8A] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <span className="font-medium">Email:</span> waliulrayhan@gmail.com
                </p>
                <p className="text-[#374151] flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#1E3A8A] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                  <span className="font-medium">Phone:</span> (+880) 1303099926
                </p>
              </div>
            </motion.div>
          </div>
          
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.9 }}
            className="mt-8 pt-6 border-t border-gray-200"
          >
            <Link 
              href="/"
              className="inline-flex items-center text-[#1E3A8A] hover:text-[#10B981] transition-colors duration-300"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Home
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}