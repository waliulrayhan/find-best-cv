"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { createPortal } from "react-dom";

export type ToastType = "success" | "error" | "info" | "warning";

export interface ToastProps {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
  onClose: (id: string) => void;
}

export const Toast = ({ id, message, type, duration = 5000, onClose }: ToastProps) => {
  const [progress, setProgress] = useState(100);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose(id);
    }, duration);
    
    const interval = setInterval(() => {
      setProgress((prev) => {
        const newProgress = prev - (100 / (duration / 100));
        return newProgress <= 0 ? 0 : newProgress;
      });
    }, 100);
    
    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, [duration, id, onClose]);
  
  const getToastStyles = () => {
    switch (type) {
      case "success":
        return {
          bg: "bg-[#10B981]/20",
          border: "border-l-4 border-[#10B981]",
          text: "text-[#0B815A]",
          icon: (
            <svg className="h-5 w-5 text-[#10B981]" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          ),
          progressColor: "bg-[#10B981]"
        };
      case "error":
        return {
          bg: "bg-red-100",
          border: "border-l-4 border-red-500",
          text: "text-red-800",
          icon: (
            <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          ),
          progressColor: "bg-red-500"
        };
      case "warning":
        return {
          bg: "bg-[#F59E0B]/20",
          border: "border-l-4 border-[#F59E0B]",
          text: "text-[#B97509]",
          icon: (
            <svg className="h-5 w-5 text-[#F59E0B]" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ),
          progressColor: "bg-[#F59E0B]"
        };
      case "info":
      default:
        return {
          bg: "bg-[#1E3A8A]/20",
          border: "border-l-4 border-[#1E3A8A]",
          text: "text-[#15296D]",
          icon: (
            <svg className="h-5 w-5 text-[#1E3A8A]" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          ),
          progressColor: "bg-[#1E3A8A]"
        };
    }
  };
  
  const styles = getToastStyles();
  
  return (
    <motion.div
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 50 }}
      className={`${styles.bg} ${styles.border} rounded-lg shadow-lg overflow-hidden mb-3 max-w-md w-full`}
    >
      <div className="p-4 flex">
        <div className="flex-shrink-0">
          {styles.icon}
        </div>
        <div className="ml-3 flex-1">
          <p className={`text-sm font-medium ${styles.text}`}>{message}</p>
        </div>
        <div className="ml-4 flex-shrink-0 flex">
          <button
            onClick={() => onClose(id)}
            className="bg-transparent rounded-md inline-flex text-gray-500 hover:text-gray-700 focus:outline-none"
          >
            <span className="sr-only">Close</span>
            <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      <div className="h-1 w-full bg-gray-200">
        <div 
          className={`h-full ${styles.progressColor}`} 
          style={{ width: `${progress}%`, transition: 'width 100ms linear' }}
        ></div>
      </div>
    </motion.div>
  );
};

export interface ToastContainerProps {
  children: React.ReactNode;
}

export const ToastContainer = ({ children }: ToastContainerProps) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!mounted) return null;
  
  return createPortal(
    <div className="fixed top-4 right-4 z-50 flex flex-col items-end">
      <AnimatePresence>{children}</AnimatePresence>
    </div>,
    document.body
  );
}; 