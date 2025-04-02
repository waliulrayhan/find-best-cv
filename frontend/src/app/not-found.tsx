import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Page Not Found</h2>
        <p className="text-gray-600 mb-4">The page you're looking for doesn't exist.</p>
        <Link 
          href="/"
          className="text-blue-600 hover:text-blue-800"
        >
          Go back home
        </Link>
      </div>
    </div>
  );
} 