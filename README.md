# CV Matcher - Smart Resume Screening Tool

A powerful AI-driven application that helps recruiters and HR professionals find the best candidates by automatically screening and ranking resumes/CVs based on job requirements.

![image](https://github.com/user-attachments/assets/725d9e17-dba0-4024-bc5e-b98c34e48632)

## 🚀 Features

- **Multi-CV Upload**: Upload multiple resumes (PDF, DOCX) at once
- **Job Description Analysis**: Upload or enter job descriptions to define requirements
- **AI-Powered Matching**: Advanced NLP algorithms to rank candidates by relevance
- **Keyword Matching**: Identify matching skills and keywords between job descriptions and CVs
- **Interactive Results**: View detailed matching results with similarity scores
- **Responsive Design**: Fully responsive UI that works on all devices
- **Email Notifications**: Contact form with email integration

## 🛠️ Technology Stack

### Frontend
- **Next.js**: React framework for building the UI
- **TypeScript**: For type-safe code
- **Tailwind CSS**: For styling and responsive design
- **Framer Motion**: For animations and transitions
- **React Icons**: For UI icons

### Backend
- **FastAPI**: Python framework for API development
- **scikit-learn**: For text processing and similarity calculation
- **NLTK**: For natural language processing
- **PyMuPDF & python-docx**: For extracting text from PDF and DOCX files
- **Nodemailer**: For email sending functionality

## 📂 Project Structure

```
cv-matcher/
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/                # Next.js app directory
│   │   │   ├── api/            # API routes
│   │   │   ├── components/     # Reusable UI components
│   │   │   ├── context/        # React context providers
│   │   │   ├── contact/        # Contact page
│   │   │   ├── upload/         # CV upload page
│   │   │   ├── results/        # Results display page
│   │   │   ├── privacy/        # Privacy policy page
│   │   │   ├── terms/          # Terms of service page
│   │   │   └── page.tsx        # Home page
│   │   └── ...
│   ├── public/                 # Static assets
│   └── ...
├── backend/                    # FastAPI backend
│   ├── main.py                 # Main FastAPI application
│   ├── text_processor.py       # NLP text processing utilities
│   └── uploads/                # Directory for uploaded files
└── README.md                   # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Node.js (v16+)
- Python (v3.8+)
- npm or yarn

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install

# Create .env.local file with email credentials
# Add the following variables:
# EMAIL_USER=your-email@gmail.com
# EMAIL_PASS=your-app-password

# Start development server
npm run dev
# or
yarn dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn scikit-learn nltk pymupdf python-docx

# Start FastAPI server
uvicorn main:app --reload
```

## 📘 Usage

1. **Home Page**: Learn about the application and its features
2. **Upload Page**: Upload job descriptions and CVs for matching
   - Supported formats: PDF and DOCX
   - Maximum file size: 10MB
3. **Results Page**: View ranked candidates with detailed matching information
   - Similarity scores between job requirements and each CV
   - Matching keywords for each candidate
   - Option to view full CV text
4. **Contact Page**: Get in touch with the team for support or inquiries

## 🔌 API Endpoints

### Backend API (FastAPI)

- **POST /match-cvs/**: Submit job description and CVs for matching
  - Input: Form data with files
  - Output: JSON with matching results
- **GET /download-file/{filename}**: Download a previously uploaded file
- **GET /health**: Health check endpoint

### Frontend API (Next.js)

- **POST /api/contact**: Submit contact form data
  - Input: JSON with name, email, subject, message
  - Output: JSON with success status and message

## ⚙️ How It Works

1. **Text Extraction**: The system extracts text from uploaded PDF and DOCX files
2. **Text Preprocessing**:
   - Converting to lowercase
   - Removing punctuation and special characters
   - Tokenizing and removing stopwords
   - Applying stemming
3. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical vectors
4. **Similarity Calculation**: Cosine similarity is calculated between the job description vector and each CV vector
5. **Ranking**: CVs are ranked based on their similarity scores in descending order
6. **Keyword Matching**: Common words between job descriptions and CVs are identified and highlighted

## 🚧 Future Improvements

- User authentication and account management
- Saved job templates and matching histories
- More advanced NLP techniques for better matching
- Integration with job boards and applicant tracking systems
- Batch processing for large-scale CV screening

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contact

For any questions or support, please use the contact form in the application or email directly at waliulrayhan@gmail.com.

