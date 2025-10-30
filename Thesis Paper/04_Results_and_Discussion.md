\section{Results and Discussion}
\label{sec:results}

This section presents the evaluation results of the proposed hybrid deep learning system for automated resume screening across training dynamics, classification performance, and component contribution analysis.

\subsection{Training Dynamics and Convergence Analysis}

Figure~\ref{fig:training_curves} presents the training and validation learning curves, demonstrating model convergence and generalization capability.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/fig1_training_validation_curves.png}
\caption{Training and Validation Learning Curves}
\label{fig:training_curves}
\end{figure}

The loss curves demonstrate smooth convergence over 12 epochs, with training and validation losses decreasing from 3.18 to 0.70 and 1.04 respectively. The final loss gap of 0.379 indicates effective learning without significant overfitting. Accuracy curves show parallel progression, reaching 85.25\% test accuracy with rapid validation improvement in epochs 5-9. The overfitting indicator reveals a final accuracy gap of -0.003, well within the 5\% threshold, confirming generalization. All metrics (precision, recall, F1-score) converge above 0.80, demonstrating balanced performance between false positives and false negatives.

\subsection{Classification Performance Analysis}

Figure~\ref{fig:confusion_matrix} shows the confusion matrix for all 24 job categories.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/fig2_confusion_matrix.png}
\caption{Confusion Matrix - Test Set Performance}
\label{fig:confusion_matrix}
\end{figure}

The confusion matrix exhibits strong diagonal dominance with 85.25\% overall accuracy across 373 test samples. Eight categories achieved perfect or near-perfect classification: HR (18/18), Healthcare (18/19), Fitness (16/16), BPO (19/19), Sales (11/11), Digital Media (11/11), Engineering (15/15), and Aviation (12/12). Minor misclassifications occur between semantically related categories (e.g., Designer with Information-Technology, Accountant with Finance), indicating logical confusion patterns rather than systematic weaknesses.

\subsection{Component Contribution Analysis}

Figure~\ref{fig:component_weights} illustrates the learned component weights through attention-based fusion.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/fig4_model_component_weights.png}
\caption{Hybrid Model Component Weights - Initial vs Learned}
\label{fig:component_weights}
\end{figure}

The weight comparison reveals significant adaptation from initial settings. BERT decreased from 40\% to 30.4\%, CNN from 25\% to 24.2\%, and LSTM from 25\% to 24.4\%. Notably, Traditional ML increased from 10\% to 21.1\%, more than doubling its contribution. This demonstrates the attention mechanism's effectiveness in learning optimal combinations, with traditional features (TF-IDF, skill counts) complementing deep learning representations effectively in the resume screening domain.

\subsection{Precision-Recall Analysis}

Figure~\ref{fig:precision_recall} shows precision-recall dynamics during training.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/fig7_precision_recall_analysis.png}
\caption{Precision-Recall Analysis}
\label{fig:precision_recall}
\end{figure}

The metrics demonstrate synchronized improvement, with precision and recall reaching 0.81. The trajectory shows steady progression from low initial performance to high final performance (0.82 recall, 0.81 precision), indicating balanced optimization without sacrificing either metric. Recall exhibits slightly faster early improvement, suggesting the model learns to identify positive cases broadly before refining classification boundaries.

\subsection{Overall Performance Summary}

Figure~\ref{fig:performance_summary} provides a comprehensive performance overview across multiple evaluation dimensions.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/fig9_performance_summary.png}
\caption{Overall Model Performance Summary}
\label{fig:performance_summary}
\end{figure}

The model achieved 85.25\% accuracy, 85.46\% precision, 85.25\% recall, and 84.56\% F1-score, all exceeding the 80\% target threshold. The per-class F1-score distribution shows strong performance with mean 0.780 and median 0.889, where 16 out of 24 categories achieve F1-scores above 0.80. The box plot analysis shows tight interquartile ranges (medians 0.80-0.85) with minimal outliers, while the radar chart confirms balanced capabilities across all evaluation dimensions.

\subsection{Production Deployment and Web Interface}

The system has been deployed as a production-ready web application at \url{https://cv-matcher-phi.vercel.app}, demonstrating practical applicability in real-world recruitment scenarios.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/web_interface_upload.png}
\caption{Web Interface - File Upload Page}
\label{fig:web_upload}
\end{figure}

Figure~\ref{fig:web_upload} shows the upload interface where recruiters can upload job descriptions and multiple CV files (PDF/DOCX, max 10MB each) with drag-and-drop functionality.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{Figures/web_interface_results.png}
\caption{CV Matching Results with Ranked Candidates}
\label{fig:web_results}
\end{figure}

Figure~\ref{fig:web_results} displays ranked candidates with match scores (32\%-62\%), matched keywords, and color-coded ratings (green/yellow/orange) for immediate visual feedback. The deployment uses Vercel's edge network, FastAPI backend, and Next.js frontend for scalable real-world recruitment automation.

\subsection{Key Findings}

The experimental results demonstrate several key achievements:
\begin{enumerate}
    \item\textbf{Strong Generalization}: 85.25\% test accuracy with train-validation gap of -0.003
    \item\textbf{Balanced Performance}: Precision (85.46\%) and recall (85.25\%) are nearly identical, avoiding bias toward false positives or negatives
    \item\textbf{Effective Fusion}: Traditional ML contribution increased from 10\% to 21.1\%, validating the hybrid approach
    \item\textbf{Robust Performance}: 16/24 categories achieved F1-scores above 0.80 with median 0.889
    \item\textbf{Production Viability}: Successfully deployed at \url{https://cv-matcher-phi.vercel.app}, processing real-world recruitment requests with intuitive web interface
\end{enumerate}

These results validate that combining BERT, CNN, LSTM with traditional ML features yields superior performance for automated resume screening, with balanced metrics and production-ready quality.
---

\begin{thebibliography}{99}

\bibitem{ref_placeholder}
References will be consolidated at the end of the complete paper.

\end{thebibliography}
