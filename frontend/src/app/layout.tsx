import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-white dark:bg-white`}
      >
        <div className="min-h-screen flex flex-col">
          <header className="bg-white dark:bg-white shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
              <div className="flex justify-between items-center">
                <a href="/" className="flex items-center space-x-3">
                  <span className="text-2xl font-bold text-blue-600 dark:text-blue-600">CV Matcher</span>
                </a>
                <nav className="flex space-x-6">
                  <a href="/" className="text-gray-700 dark:text-gray-700 hover:text-blue-600 dark:hover:text-blue-600">Home</a>
                  <a href="/upload" className="text-gray-700 dark:text-gray-700 hover:text-blue-600 dark:hover:text-blue-600">Upload</a>
                </nav>
              </div>
            </div>
          </header>
          <main className="flex-grow">
            {children}
          </main>
          <footer className="bg-white dark:bg-white shadow-inner">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
              <p className="text-center text-gray-500 dark:text-gray-500 text-sm">
                © {new Date().getFullYear()} CV Matcher. All rights reserved.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
