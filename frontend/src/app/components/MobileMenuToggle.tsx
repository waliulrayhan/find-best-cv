"use client";

import { useState } from 'react';

export default function MobileMenuToggle() {
  const [isOpen, setIsOpen] = useState(false);
  
  const toggleMenu = () => {
    setIsOpen(!isOpen);
    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenu) {
      mobileMenu.classList.toggle('hidden');
    }
  };

  return (
    <div className="md:hidden flex items-center">
      <button 
        type="button" 
        className="text-gray-500 hover:text-[#1E3A8A] focus:outline-none"
        aria-label="Toggle menu"
        onClick={toggleMenu}
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
    </div>
  );
} 