# CV Screening Tool - Smart Resume Matching with AI

A powerful AI-driven CV screening application that uses hybrid deep learning (BERT + CNN + LSTM) to automatically screen and rank resumes based on job requirements. **Optimized for NVIDIA GPUs** with CUDA support.

## 🚀 Features

- **GPU-Accelerated Training**: Optimized for NVIDIA GPUs (RTX 3050 and above)
- **Hybrid Deep Learning Model**: Combines BERT, CNN, LSTM, and traditional ML
- **Multi-CV Upload**: Upload multiple resumes (PDF, DOCX) at once
- **Job Description Analysis**: Intelligent matching based on semantic understanding
- **High Accuracy**: Optimized for best performance on CV-job matching
- **Mixed Precision Training**: Faster training with FP16 on compatible GPUs
- **Interactive Web UI**: Modern Next.js frontend with real-time results
- **RESTful API**: FastAPI backend for easy integration

## 🛠️ Technology Stack

### Machine Learning
- **PyTorch 2.0+**: Deep learning framework with CUDA 12.1 support
- **Transformers**: DistilBERT for semantic understanding
- **scikit-learn**: Traditional ML features and metrics
- **NLTK**: Natural language preprocessing

### Backend
- **FastAPI**: High-performance Python API framework
- **PyMuPDF & python-docx**: Document text extraction
- **Uvicorn**: ASGI server

### Frontend
- **Next.js 14**: React framework with TypeScript
- **Tailwind CSS**: Modern styling
- **Framer Motion**: Smooth animations

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or AMD Ryzen 5
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with 4GB VRAM (RTX 3050 or better)
- **Storage**: 10 GB free space

### Recommended
- **CPU**: Intel Core i7 or AMD Ryzen 7
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX 3060 or better (6GB+ VRAM)
- **Storage**: 20 GB SSD

### GPU Support
- CUDA 12.1 compatible NVIDIA GPU
- cuDNN 9.0+
- Latest NVIDIA drivers

## 📂 Project Structure

```
cv-screening-tool/
├── backend/                    # FastAPI backend
│   ├── main.py                 # Main API server
│   ├── text_processor.py       # Text processing utilities
│   └── requirements.txt
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/                # Next.js pages and API routes
│   │   └── components/         # React components
│   └── package.json
├── src/                        # ML model source code
│   ├── data/                   # Data processing modules
│   │   └── data_processor.py
│   ├── models/                 # Model implementations
│   │   ├── hybrid_model.py     # Hybrid model architecture
│   │   ├── trainer.py          # Training pipeline
│   │   └── predictor.py        # Inference module
│   └── utils/                  # Utility functions
├── config/                     # Configuration files
│   └── config.py               # Model & training configs
├── data/                       # Datasets
│   ├── raw/                    # Raw CV data
│   └── processed/              # Processed features
├── models/                     # Trained models
│   ├── hybrid_model.pth        # Trained weights
│   ├── tokenizer/              # BERT tokenizer
│   └── model_metadata.json     # Model metadata
├── experiments/                # Training logs and results
│   ├── logs/                   # Training logs
│   └── results/                # Experiment results
├── check_gpu.py                # GPU verification script
├── train_model.py              # Training script
├── requirements.txt            # Python dependencies
└── docker-compose.yml          # Docker setup
```

## 🔧 Installation & Setup

### Prerequisites

1. **Install NVIDIA Drivers**
   - Download and install the latest drivers from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

2. **Install CUDA Toolkit 12.1**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

3. **Install Python 3.8+**
   - Download from [Python.org](https://www.python.org/downloads/)

4. **Install Node.js 18+** (for frontend)
   - Download from [Node.js website](https://nodejs.org/)

### Step 1: Verify GPU Setup

```bash
# Clone the repository
git clone https://github.com/waliulrayhan/find-best-cv.git
cd cv-screening-tool

# Check GPU availability
python check_gpu.py
```

Expected output:
```
✅ GPU IS AVAILABLE AND WORKING!
Your model will use GPU for training and inference.
GPU Device: NVIDIA GeForce RTX 3050 Laptop GPU
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 3: Prepare Data (Optional - for training)

If you want to train the model from scratch:

```bash
# Place your resume dataset in data/raw/Resume.csv
# The dataset should have columns: Resume (text) and Category (label)

# Process the data
python -c "from src.data.data_processor import EnhancedDataProcessor; EnhancedDataProcessor().process_dataset()"
```

### Step 4: Train the Model (Optional)

```bash
# Train with GPU acceleration
python train_model.py
```

Training progress will be displayed with:
- Real-time loss and accuracy metrics
- GPU memory usage
- Estimated time remaining
- Best model checkpoints saved automatically

### Step 5: Run the Backend API

```bash
# Navigate to backend directory
cd backend

# Install backend dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

API will be available at `http://localhost:8000`

### Step 6: Run the Frontend (Optional)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 🎯 Usage

### Quick Test

```bash
# Verify GPU is being used
python check_gpu.py

# Test model inference
python -c "from src.models.predictor import HybridModelPredictor; predictor = HybridModelPredictor(); print('Model loaded successfully!')"
```

### API Endpoints

**1. Health Check**
```bash
GET http://localhost:8000/
```

**2. Match Resume to Job**
```bash
POST http://localhost:8000/match
Content-Type: multipart/form-data

Body:
- resume: PDF/DOCX file
- job_description: text
```

**3. Batch Process Multiple Resumes**
```bash
POST http://localhost:8000/batch-match
Content-Type: multipart/form-data

Body:
- resumes: multiple PDF/DOCX files
- job_description: text
```

### Web Interface

1. Open browser and navigate to `http://localhost:3000`
2. Upload resume(s) (PDF or DOCX format)
3. Enter or upload job description
4. Click "Find Best Match"
5. View ranked results with matching scores

## ⚙️ Configuration

### GPU Settings

Edit `config/config.py` to optimize for your GPU:

```python
# For RTX 3050 (4GB VRAM)
TRAINING_CONFIG = {
    "batch_size": 8,  # Adjust based on your VRAM
    "use_mixed_precision": True,  # Enable FP16
    ...
}

HARDWARE_CONFIG = {
    "use_cuda": True,  # Enable GPU
    "num_workers": 2,  # Data loading workers
    ...
}
```

### Model Settings

```python
# Adjust model complexity
BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",  # Smaller BERT
    "max_length": 512,
    "freeze_bert": False,  # Fine-tune BERT
    ...
}
```

## 📊 Performance

### Training Results

Performance metrics will vary based on your specific dataset and training configuration. The model is optimized to achieve the best possible accuracy while preventing overfitting.

Example metrics from a training run:
- **Accuracy**: 83-85%
- **Precision**: 82-84%
- **Recall**: 83-85%
- **F1-Score**: 82-84%

### Inference Speed (RTX 3050)

- Single CV: ~50ms
- Batch (10 CVs): ~300ms
- GPU Memory Usage: ~2.5GB

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory Error

Reduce batch size in `config/config.py`:
```python
TRAINING_CONFIG = {
    "batch_size": 4,  # Reduce from 8
    "accumulation_steps": 4,  # Increase to simulate larger batch
}
```

### Model Not Found

```bash
# Download pre-trained model or train from scratch
python train_model.py
```

## 📝 Development

### Project Structure

```bash
# Clone the repository
git clone https://github.com/waliulrayhan/find-best-cv.git
cd find-best-cv

# Build and run using Docker Compose
docker-compose up --build

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

✅ Make sure Docker & Docker Compose are installed on your system.
```

### 🧑‍💻 Option 2: Manual Setup

### Prerequisites
- Node.js (v16+)

### Project Layout

```
src/
├── data/              # Data processing and feature extraction
├── models/            # Model architecture, training, inference
└── utils/             # Helper utilities
```

### Key Files

- `check_gpu.py` - Verify GPU setup
- `train_model.py` - Train the model
- `config/config.py` - All configurations
- `backend/main.py` - API server
- `frontend/src/app/page.tsx` - Web UI

## 🚀 Model Architecture

### Hybrid Approach

The model combines multiple architectures for optimal performance:

1. **BERT Component** (40% weight)
   - DistilBERT for semantic understanding
   - Fine-tuned on CV-job matching task

2. **CNN Component** (25% weight)
   - Multiple filter sizes (3, 4, 5)
   - Extracts local patterns and phrases

3. **LSTM Component** (25% weight)
   - Bidirectional LSTM
   - Captures sequential dependencies

4. **Traditional ML** (10% weight)
   - TF-IDF features
   - Skill and experience extraction

### Training Pipeline

1. Data preprocessing and augmentation
2. Feature extraction (BERT embeddings, TF-IDF)
3. Hybrid model training with:
   - Mixed precision (FP16)
   - Gradient accumulation
   - Early stopping
   - Learning rate scheduling
4. Model evaluation and validation
5. Export for production use

## � Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Services will be available at:
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## 📊 Monitoring

### Training Logs

```bash
# View training logs
tail -f experiments/logs/training.log

# View experiment results
ls experiments/results/
```

### GPU Monitoring

```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Or use the built-in monitoring
python check_gpu.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Waliul Rayhan** - [@waliulrayhan](https://github.com/waliulrayhan)

## 🙏 Acknowledgments

- DistilBERT by HuggingFace Transformers
- PyTorch team for excellent deep learning framework
- FastAPI for the modern API framework
- Next.js team for the amazing React framework

## � Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/waliulrayhan/find-best-cv/issues)
- Email: waliulrayhan@gmail.com

## 📈 Performance Tips

### For Training
- Use batch size 8-16 for RTX 3050 (4GB VRAM)
- Enable mixed precision training
- Use gradient accumulation if OOM errors occur
- Monitor GPU temperature during long training sessions

### For Inference
- Batch multiple CVs for faster processing
- Use GPU for batches > 5 CVs
- Use CPU for single CV quick checks
- Cache tokenizer and model for multiple inferences

---

Made with ❤️ using PyTorch and CUDA

