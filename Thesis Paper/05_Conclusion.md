\section{Conclusion}
\label{sec:conclusion}

This research presented a novel hybrid deep learning architecture for automated resume screening that integrates DistilBERT, CNN, LSTM, and traditional ML features through attention-based fusion. The system achieved 85.25\% test accuracy, 85.46\% precision, 85.25\% recall, and 84.56\% F1-score across 24 job categories with a train-validation gap of only 0.34\%, demonstrating strong generalization without overfitting. The attention mechanism learned optimal component weights, with traditional ML contribution increasing from 10\% to 21.1\%, validating the complementary nature of deep learning and traditional approaches. With 16 out of 24 categories achieving F1-scores above 0.80 and median F1-score of 0.889, the model demonstrated robust cross-category performance. The production-ready implementation features FastAPI backend, Next.js frontend, and Docker deployment, reducing manual screening time from hours to minutes. Resource-efficient training on consumer GPU (4GB VRAM) using mixed precision and gradient accumulation converged within 12 epochs, democratizing access to advanced AI technology for organizations.

Future work should address several limitations including the relatively small dataset (2,483 samples), English-language restriction, and lack of explainability mechanisms. Promising research directions include integrating explainable AI through attention visualization and LIME/SHAP techniques, extending multilingual support using mBERT or XLM-RoBERTa, implementing active learning with human-in-the-loop feedback, developing specialized NER models for fine-grained skill extraction, investigating fairness and bias mitigation for equitable evaluation, and enabling dynamic job market adaptation through continual learning. This work demonstrates that hybrid architectures combining multiple complementary components achieve superior performance for automated resume screening compared to single-model approaches. As recruitment technology markets expand and application volumes increase, intelligent automation solutions become essential for competitive hiring practices, and this research advances the field by providing both theoretical insights into hybrid neural architectures and practical solutions for real-world recruitment challenges.

---

\begin{thebibliography}{99}

\bibitem{ref_placeholder}
References will be consolidated at the end of the complete paper.

\end{thebibliography}
