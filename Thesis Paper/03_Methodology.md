\section{Methodology}
\label{sec:methodology}

This section presents the methodology employed in developing the hybrid deep learning system for automated resume screening. The system architecture integrates data preprocessing, hybrid model inference, training optimization, and deployment infrastructure as illustrated in Figure~\ref{fig:system_architecture}. Algorithm~\ref{alg:overall_workflow} describes the complete workflow of the proposed system.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/system_architecture.png}
\caption{Complete System Architecture: Data Input \& Preprocessing, Hybrid Model Architecture, Inference \& Deployment, and Output \& Results}
\label{fig:system_architecture}
\end{figure}

\begin{algorithm}[htbp]
\caption{Overall System Workflow}
\label{alg:overall_workflow}
\begin{algorithmic}[1]
\Require Resume dataset $D$, Job description $J$
\Ensure Ranked candidate list $R$
\State \textbf{Phase 1: Data Preprocessing}
\State Extract text from PDF/DOCX files using PyMuPDF and python-docx
\State Clean text: lowercase, remove URLs/emails/phones, normalize whitespace
\State Apply NLP: tokenization, stopword removal, lemmatization (NLTK)
\State Extract features: TF-IDF (10k features), skills (6 categories), statistics
\State Split data: 70\% train, 15\% validation, 15\% test (stratified)
\State
\State \textbf{Phase 2: Model Training}
\For{each epoch $e = 1$ to $E_{max}$}
    \For{each batch $(x, y)$ in $D_{train}$}
        \State Tokenize input for BERT (max 512 tokens)
        \State $f_{BERT} \gets$ DistilBERT component forward pass
        \State $f_{CNN} \gets$ Multi-kernel CNN forward pass
        \State $f_{LSTM} \gets$ BiLSTM with attention forward pass
        \State $f_{traditional} \gets$ TF-IDF + handcrafted features
        \State $f_{fused} \gets$ Attention-based fusion of all components
        \State $\hat{y} \gets$ Classification head (softmax)
        \State Compute loss: $\mathcal{L}_{CE} = -\sum y \log(\hat{y})$
        \State Backward pass with mixed precision (FP16)
        \State Update weights using AdamW with gradient clipping
    \EndFor
    \State Evaluate on $D_{val}$, save best model, apply early stopping
\EndFor
\State
\State \textbf{Phase 3: Deployment \& Inference}
\State Load trained model and preprocessors
\State Accept CV files and job description via FastAPI endpoint
\State Preprocess and extract features from input CVs
\State Perform batch inference to predict job categories
\State Compute cosine similarity between CVs and job description
\State $R \gets$ Rank candidates by weighted similarity score
\State \Return $R$ with confidence scores and match percentages
\end{algorithmic}
\end{algorithm}

\subsection{Data Collection and Preprocessing}

The system is trained on a publicly available resume dataset~\cite{kaggle_resume_dataset} containing 2,483 resume samples across 24 job categories (HR, Designer, Information Technology, Teacher, etc.) with a stratified split ratio of 70\% training, 15\% validation, and 15\% testing.

The preprocessing pipeline consists of four stages: (1) text extraction from PDF/DOCX using PyMuPDF and python-docx, (2) text cleaning and normalization including lowercase conversion, URL/email/phone removal, and whitespace normalization, (3) NLP preprocessing using NLTK for tokenization, stopword removal, and lemmatization, and (4) feature extraction including TF-IDF vectorization (max 10,000 features, n-gram range 1-3), skill pattern matching across six categories (programming languages, frameworks, databases, cloud platforms, tools, soft skills), and statistical features extraction (text length, word count, skill counts). Stratified sampling ensures proportional class representation across splits.

\subsection{Hybrid Model Architecture}

The hybrid model integrates four complementary components through attention-based fusion. The \textbf{BERT component} uses DistilBERT~\cite{sanh2019distilbert} (768-dim, 6 transformer layers, 12 attention heads) for semantic understanding, transforming the CLS token through MLP layers (768→384→192) to produce $f_{BERT} \in \mathbb{R}^{192}$. The \textbf{CNN component} employs multi-kernel convolutions (filter sizes 3,4,5 with 100 filters each) for local n-gram pattern recognition, applying global max pooling to generate $f_{CNN} \in \mathbb{R}^{150}$. The \textbf{LSTM component} utilizes bidirectional LSTM (2 layers, 256 hidden units) with self-attention mechanism to capture sequential dependencies, producing $f_{LSTM} \in \mathbb{R}^{128}$. The \textbf{Traditional ML component} combines TF-IDF features with handcrafted features (skill counts, experience levels).

The attention-based fusion mechanism dynamically weights component contributions:
\begin{equation}
f_{fused} = \sum_{i=1}^{4} w_i f_i, \quad w_i = \frac{\exp(\alpha_i)}{\sum_{j=1}^{4} \exp(\alpha_j)}, \quad \sum_{i=1}^{4} w_i = 1
\end{equation}
where initial weights are $w_{BERT}=0.4, w_{CNN}=0.25, w_{LSTM}=0.25, w_{traditional}=0.1$. The fused features pass through classification layers with batch normalization and dropout (0.3, 0.2) to produce final predictions:
\begin{equation}
\hat{y} = \text{Softmax}(\text{Linear}_{24}(\text{Dropout}(\text{ReLU}(\text{BatchNorm}(\text{Linear}_{256}(f_{fused}))))))
\end{equation}

\subsection{Training Strategy and Optimization}

The model is optimized using cross-entropy loss $\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_i^c \log(\hat{y}_i^c)$ with AdamW optimizer. Table~\ref{tab:training_params} summarizes the training hyperparameters and optimization strategies employed to achieve robust model performance while preventing overfitting.

\begin{table}[htbp]
\centering
\caption{Training Hyperparameters and Optimization Configuration}
\label{tab:training_params}
\begin{tabular}{|l|l|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
\multicolumn{2}{|c|}{\textbf{Optimizer Configuration}} \\
\hline
Optimizer & AdamW \\
Learning Rate ($\alpha$) & $2 \times 10^{-5}$ \\
Weight Decay ($\lambda$) & 0.05 \\
Beta 1 ($\beta_1$) & 0.9 \\
Beta 2 ($\beta_2$) & 0.999 \\
Epsilon ($\epsilon$) & $10^{-8}$ \\
\hline
\multicolumn{2}{|c|}{\textbf{Learning Rate Scheduling}} \\
\hline
Schedule Type & Linear Warmup + Linear Decay \\
Warmup Steps & 500 \\
\hline
\multicolumn{2}{|c|}{\textbf{Regularization Techniques}} \\
\hline
Dropout (Embedding) & 0.2 \\
Dropout (BERT Classifier) & 0.4 \\
Dropout (CNN/LSTM) & 0.5 / 0.4 \\
Dropout (Fusion Layers) & 0.3, 0.2 \\
Gradient Clipping ($\tau$) & 1.0 \\
Batch Normalization & Applied \\
Early Stopping Patience & 3 epochs \\
\hline
\multicolumn{2}{|c|}{\textbf{Training Configuration}} \\
\hline
Batch Size & 8 \\
Gradient Accumulation Steps & 2 \\
Effective Batch Size & 16 \\
Max Epochs & 20 \\
Mixed Precision (AMP) & FP16 \\
Hardware & NVIDIA RTX 3050 (4GB VRAM) \\
\hline
\end{tabular}
\end{table}

The training procedure iterates through epochs, performing forward-backward passes with gradient accumulation, evaluating on validation set, saving best checkpoints based on validation loss, and applying early stopping when validation loss plateaus for three consecutive epochs.

\subsection{System Deployment}

The system is deployed using a microservices architecture with Docker containerization. The \textbf{backend} uses FastAPI RESTful API (endpoint: POST /match-cvs) that accepts job descriptions and CV files (PDF/DOCX), extracts text using PyMuPDF and python-docx, preprocesses and tokenizes text, performs batch inference, computes cosine similarity scores with job descriptions, and ranks candidates. The \textbf{frontend} is built with Next.js featuring drag-and-drop file upload, real-time processing indicators, interactive results tables with sorting/filtering, match score visualizations, and export capabilities. Docker containers (Python 3.9 for backend on port 8000, Node.js 18 for frontend on port 3000) are orchestrated using docker-compose for portable deployment. Inference optimization includes batch processing, FP16 quantization, caching of vectorizers/tokenizers, asynchronous processing, and GPU/CPU auto-detection.

---

\begin{thebibliography}{99}

\bibitem{kaggle_resume_dataset}
S. Bhawal, ``Resume Dataset,'' Kaggle, 2022. [Online]. Available: \url{https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset}. [Accessed: Oct. 30, 2025]

\end{thebibliography}
