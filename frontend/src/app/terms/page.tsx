"use client"

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

export default function TermsPage() {
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </motion.div>
            <h1 className="text-3xl font-bold text-[#374151] flex-1">Terms of Service</h1>
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
                Welcome to CV Matcher. These terms and conditions outline the rules and regulations for the use of our website and services.
              </p>
              <p className="text-[#374151] mb-4">
                By accessing this website, we assume you accept these terms and conditions in full. Do not continue to use CV Matcher if you do not accept all of the terms and conditions stated on this page.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">2</span>
                License to Use
              </h2>
              <p className="text-[#374151] mb-4">
                Unless otherwise stated, CV Matcher and/or its licensors own the intellectual property rights for all material on CV Matcher. All intellectual property rights are reserved.
              </p>
              <p className="text-[#374151] mb-4">
                You may view and/or print pages from the website for your own personal use subject to restrictions set in these terms and conditions.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">3</span>
                Restrictions
              </h2>
              <p className="text-[#374151] mb-4">
                You are specifically restricted from all of the following:
              </p>
              <ul className="list-none pl-5 text-[#374151] mb-4 space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Publishing any website material in any other media.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Selling, sublicensing and/or otherwise commercializing any website material.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Using this website in any way that is or may be damaging to this website.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Using this website in any way that impacts user access to this website.
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#F59E0B] mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  Using this website contrary to applicable laws and regulations, or in any way may cause harm to the website, or to any person or business entity.
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
                Your Content
              </h2>
              <p className="text-[#374151] mb-4">
                In these terms and conditions, &apos;Your Content&apos; shall mean any audio, video, text, images or other material you choose to display on this website. By displaying Your Content, you grant CV Matcher a non-exclusive, worldwide, irrevocable, royalty-free, sublicensable license to use, reproduce, adapt, publish, translate and distribute it in any and all media.
              </p>
              <p className="text-[#374151] mb-4">
                Your Content must be your own and must not be infringing on any third party&apos;s rights. CV Matcher reserves the right to remove any of Your Content from this website at any time without notice.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">5</span>
                No Warranties
              </h2>
              <div className="bg-[#F8FAFC] p-4 rounded-lg border-l-2 border-[#10B981]">
                <p className="text-[#374151] mb-0">
                  This website is provided &quot;as is,&quot; with all faults, and CV Matcher makes no express or implied representations or warranties, of any kind related to this website or the materials contained on this website.
                </p>
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">6</span>
                Limitation of Liability
              </h2>
              <p className="text-[#374151] mb-4">
                In no event shall CV Matcher, nor any of its officers, directors and employees, be held liable for anything arising out of or in any way connected with your use of this website, whether such liability is under contract, tort or otherwise.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
            >
              <h2 className="text-xl font-semibold text-[#1E3A8A] mt-8 mb-4 flex items-center">
                <span className="bg-[#1E3A8A] text-white rounded-full w-7 h-7 inline-flex items-center justify-center mr-2">7</span>
                Governing Law & Jurisdiction
              </h2>
              <p className="text-[#374151] mb-4">
                These terms will be governed by and interpreted in accordance with the laws of the country/state where CV Matcher is based, and you submit to the non-exclusive jurisdiction of the state and federal courts located there for the resolution of any disputes.
              </p>
            </motion.div>
          </div>
          
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.5 }}
            className="mt-8 pt-6 border-t border-gray-200"
          >
            <Link 
              href="/"
              className="inline-flex items-center text-[#1E3A8A] hover:text-[#10B981] transition-colors duration-300 bg-white hover:bg-[#F8FAFC] px-4 py-2 rounded-lg shadow-sm"
            >
              <motion.div
                whileHover={{ x: -3 }}
                transition={{ duration: 0.2 }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </motion.div>
              Back to Home
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
} 