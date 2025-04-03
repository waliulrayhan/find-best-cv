import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ErrorBoundary } from "./components/ErrorBoundary";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CV Screening Tool",
  description: "Match the best candidates to your job descriptions using AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#F8FAFC] text-[#374151]`}
      >
        <ErrorBoundary>
          <div className="min-h-screen flex flex-col">
            <header className="bg-white shadow-sm sticky top-0 z-10 animate-fadeDown">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex justify-between items-center">
                  <Link href="/" className="flex items-center space-x-3 transition-transform hover:scale-105">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-800 rounded-lg flex items-center justify-center shadow-md animate-float">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-800 animate-shimmer">CV Matcher</span>
                  </Link>
                  <nav className="hidden md:flex space-x-8">
                    <Link href="/" className="text-[#374151] hover:text-blue-600 transition-colors duration-300 font-medium flex items-center gap-1 hover:scale-105 transform transition-transform">
                      Home
                    </Link>
                    <Link href="/upload" className="text-[#374151] hover:text-blue-600 transition-colors duration-300 font-medium flex items-center gap-1 hover:scale-105 transform transition-transform">
                      Upload CVs
                    </Link>
                    <Link href="/contact" className="text-[#374151] hover:text-blue-600 transition-colors duration-300 font-medium flex items-center gap-1 hover:scale-105 transform transition-transform">
                      Contact
                    </Link>
                  </nav>
                  <Link 
                    href="/upload" 
                    className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white px-5 py-2.5 rounded-full font-medium transition-all duration-300 hover:shadow-lg hover:shadow-blue-300/50 flex items-center gap-2 group animate-pulse-slow hover:animate-none"
                  >
                    <span>Get Started</span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </Link>
                </div>
              </div>
            </header>
            <main className="flex-grow animate-fadeIn">
              {children}
            </main>
            <footer className="bg-white shadow-inner mt-12 animate-fadeUp">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  <div className="flex flex-col space-y-4 animate-slideInLeft">
                    <Link href="/" className="flex items-center space-x-2">
                      <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-800 rounded-lg flex items-center justify-center shadow-sm animate-float">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-800 animate-shimmer">CV Matcher</span>
                    </Link>
                    <p className="text-sm text-gray-500">
                      AI-powered CV screening to find the perfect match for your job openings.
                    </p>
                  </div>
                  <div className="flex flex-col space-y-4 animate-slideInUp">
                    <h3 className="font-semibold text-[#374151]">Legal</h3>
                    <div className="flex flex-col space-y-2">
                      <Link href="/privacy" className="text-sm text-gray-500 hover:text-blue-600 transition-colors duration-300 flex items-center gap-1 hover:translate-x-1 transform transition-transform">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                        Privacy Policy
                      </Link>
                      <Link href="/terms" className="text-sm text-gray-500 hover:text-blue-600 transition-colors duration-300 flex items-center gap-1 hover:translate-x-1 transform transition-transform">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Terms of Service
                      </Link>
                      <Link href="/contact" className="text-sm text-gray-500 hover:text-blue-600 transition-colors duration-300 flex items-center gap-1 hover:translate-x-1 transform transition-transform">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        Contact Us
                      </Link>
                    </div>
                  </div>
                  <div className="flex flex-col space-y-4 animate-slideInRight">
                    <h3 className="font-semibold text-[#374151]">Connect with us</h3>
                    <div className="flex space-x-4">
                      <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-blue-500 transition-colors duration-300 transform hover:scale-110 hover:rotate-6">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84"></path>
                        </svg>
                      </a>
                      <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-blue-700 transition-colors duration-300 transform hover:scale-110 hover:rotate-6">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"></path>
                        </svg>
                      </a>
                      <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-900 transition-colors duration-300 transform hover:scale-110 hover:rotate-6">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path>
                        </svg>
                      </a>
                    </div>
                  </div>
                </div>
                <div className="mt-8 pt-8 border-t border-gray-200">
                  <p className="text-center text-gray-500 text-sm animate-fadeIn">
                    © {new Date().getFullYear()} CV Matcher. All rights reserved.
                  </p>
                </div>
              </div>
            </footer>
          </div>
        </ErrorBoundary>
      </body>
    </html>
  );
}
