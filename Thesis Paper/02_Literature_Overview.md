\section{Literature Overview}
\label{sec:literature}

The automation of resume–job description matching has evolved from traditional keyword-based systems to advanced deep learning frameworks. This section reviews existing approaches in automated resume screening.

\subsection{Traditional and NLP-Based Approaches}

Early resume screening systems relied on keyword extraction and similarity metrics such as Jaccard or cosine similarity. Saatçı et al.~\cite{saatci2024resume} implemented an NLP-based system using Jaccard similarity for matching candidate profiles with job categories. Kumar et al.~\cite{kumar2025resume} developed a pipeline using TF–IDF vectorization and cosine similarity for candidate ranking. However, these approaches struggled to capture semantic meaning and contextual relationships.

\subsection{Machine Learning and Deep Learning Approaches}

Machine learning enabled more robust matching through learned representations. Rojas-Galeano and Posada~\cite{rojas2022bibliometric} conducted a bibliometric study showing rapid growth in AI-based recruitment systems post-2016. Bian et al.~\cite{bian2020learning} proposed a Multi-View Co-Teaching Network combining text-based and relational-view matching to handle sparse data. Jiang et al.~\cite{jiang2020learning} integrated semantic entity extraction with LSTM-based behavioral modeling, demonstrating that combining explicit and implicit representations enhances person–job fit prediction. Modak et al.~\cite{modak2024review} identified key challenges including limited datasets, class imbalance, and lack of interpretability.

\subsection{Transformer-Based Models}

Transformer architectures revolutionized text matching through contextual embeddings. Vaishampayan et al.~\cite{vaishampayan2025human} found low correlation between human recruiters and GPT-4 evaluations, indicating limitations in replicating human judgment. Khan et al.~\cite{khan2025comparison} compared BERT, Gemini, and LLaMA 3.1, finding that LLaMA 3.1 achieved superior performance after domain-specific fine-tuning.

\subsection{Research Gaps}

Despite advances, several gaps remain: (1) lack of comprehensive hybrid systems integrating BERT, CNN, and LSTM with attention-based fusion, (2) insufficient analysis of overfitting prevention and generalization, (3) limited production-ready implementations, and (4) poor per-class performance across diverse job categories. Our work addresses these gaps through a hybrid architecture combining DistilBERT, CNN, LSTM, and traditional ML features with attention-based fusion and comprehensive regularization strategies.

---

## REFERENCES

```bibtex
@article{modak2024review,
  author    = {A. Modak and P. Shinde and A. Tiwari and S. Nalamwar},
  title     = {A Review of Resume Analysis and Job Description Matching Using Machine Learning},
  journal   = {International Journal of Recent Innovation Trends in Computing and Communication},
  volume    = {12},
  number    = {2},
  pages     = {247--250},
  year      = {2024}
}

@article{kumar2025resume,
  author    = {G. S. Kumar and K. Varshitha and K. Keerthi and N. Vikesh},
  title     = {Resume Screening Automation with NLP Techniques},
  journal   = {International Journal of Innovative Science and Research Technology},
  volume    = {10},
  number    = {5},
  year      = {2025},
  month     = {May}
}

@article{rojas2022bibliometric,
  author    = {S. Rojas-Galeano and J. Posada},
  title     = {A Bibliometric Perspective on AI Research for Job-Résumé Matching},
  journal   = {The Scientific World Journal},
  pages     = {1--18},
  year      = {2022}
}

@article{bian2020learning,
  author    = {S. Bian and W. Zhao and S. Song and T. Shi and L. Cao},
  title     = {Learning to Match Jobs with Resumes from Sparse Interaction Data Using Multi-View Co-Teaching Network},
  howpublished = {arXiv preprint arXiv:2009.13299},
  year      = {2020}
}

@article{jiang2020learning,
  author    = {J. Jiang and B. Chen and B. Fu and M. Long},
  title     = {Learning Effective Representations for Person-Job Fit by Feature Fusion},
  howpublished = {arXiv preprint arXiv:2006.07017},
  year      = {2020}
}

@article{saatci2024resume,
  author    = {M. Saatçı and R. Kaya and R. Ünlü},
  title     = {Resume Screening with Natural Language Processing (NLP)},
  journal   = {Alphanumeric Journal},
  volume    = {12},
  number    = {2},
  pages     = {121--140},
  year      = {2024}
}

@inproceedings{vaishampayan2025human,
  author    = {S. Vaishampayan and F. Hashemi and P. Shenoy and M. Valizadeh},
  title     = {Human and LLM-Based Resume Matching: An Observational Study},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2025},
  pages     = {4808--4823},
  year      = {2025}
}

@article{khan2025comparison,
  author    = {S. Khan and A. Ahmad and M. Hassan and R. Kumar},
  title     = {Comparison of Models for Resume-JD Matching: BERT, Gemini, and LLaMA 3.1},
  journal   = {IOSR Journal of Computer Engineering},
  volume    = {27},
  number    = {2},
  series    = {5},
  pages     = {1--10},
  month     = {March--April},
  year      = {2025}
}
```
