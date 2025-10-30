\section{Introduction}
\label{sec:introduction}

Automated resume screening has become critical as organizations face hundreds to thousands of applications per position, making manual review time-consuming and prone to human biases~\cite{rojas2022bibliometric}. Traditional keyword-matching systems fail to capture semantic context, resulting in high false-positive rates~\cite{faliagka2012integrated}. With the recruitment software market projected to reach \$3.85 billion by 2028~\cite{grandview2021recruitment}, intelligent automation solutions are urgently needed.

While Natural Language Processing has advanced through transformer-based architectures like BERT~\cite{devlin2019bert}, CNNs~\cite{kim2014convolutional}, and LSTMs~\cite{hochreiter1997long}, existing resume screening systems predominantly employ single-model architectures capturing only specific textual aspects~\cite{roy2020machine,ramos2003using}. This reveals a critical research gap: the absence of comprehensive hybrid systems that synergistically combine multiple neural architectures while maintaining computational efficiency and preventing overfitting.

This paper presents a novel hybrid deep learning architecture integrating four complementary components: DistilBERT for semantic understanding~\cite{sanh2019distilbert}, multi-kernel CNN for pattern recognition~\cite{zhang2017sensitivity}, bidirectional LSTM with attention for sequential modeling~\cite{zhou2016attention}, and traditional ML features. Components are fused through a learnable attention mechanism~\cite{vaswani2017attention} and trained on 2,483 resumes across 24 job categories using mixed-precision training~\cite{micikevicius2018mixed}, dropout, weight decay, and early stopping.

Our contributions include: (1) a novel hybrid architecture achieving 85.25\% test accuracy with minimal overfitting (0.34\% train-validation gap), (2) comprehensive evaluation demonstrating 85.46\% precision and 84.56\% F1-score across 24 categories, (3) production-ready implementation with GPU-optimized training, FastAPI backend~\cite{fastapi2018}, and Docker deployment~\cite{merkel2014docker}, and (4) practical insights for training multi-component models on consumer-grade hardware. This system advances automated recruitment technology through both theoretical contributions to hybrid neural architectures and practical solutions for real-world hiring challenges.

The rest of this research paper is organized as follows: Section II presents a literature overview of existing approaches in automated resume screening and hybrid deep learning architectures. Section III describes the proposed methodology, including the hybrid model architecture with BERT, CNN, LSTM components, attention-based fusion mechanism, dataset preparation, and training strategy. The results obtained by this research along with their discussion are presented in Section IV. Finally, Section V concludes this work and outlines future research directions.

---

## REFERENCES

```bibtex
@article{rojas2022bibliometric,
  author  = {Sergio Rojas-Galeano},
  title   = {A Bibliometric Perspective on AI Research for Job-Recruitment: Challenges and Opportunities},
  journal = {The Scientific World Journal},
  year    = {2022},
  volume  = {2022},
  doi     = {10.1155/2022/4183729}
}

@article{faliagka2012integrated,
  author  = {Evanthia Faliagka and Athanasios Tsakalidis and Giannis Tzimas},
  title   = {An Integrated e-Recruitment System for Automated Personality Mining and Applicant Ranking},
  journal = {Internet Research},
  year    = {2012},
  volume  = {22},
  number  = {5},
  pages   = {551--568},
  doi     = {10.1108/10662241211271545}
}

@misc{grandview2021recruitment,
  author       = {{Grand View Research}},
  title        = {Recruitment Software Market Size, Share \& Trends Analysis Report By Deployment, By Enterprise Size, By End Use, By Region, And Segment Forecasts, 2021-2028},
  year         = {2021},
  howpublished = {Market Analysis Report}
}

@article{qasem2021automatic,
  author  = {Mohammed H. Qasem and Nizar Obeid and Amjad Hudaib and Haneen Almashaqbeh and Mamoun Al Haj},
  title   = {Automatic Deep Learning Approach for Resume Parsing},
  journal = {International Journal of Advanced Computer Science and Applications},
  year    = {2021},
  volume  = {12},
  number  = {8},
  pages   = {384--391},
  doi     = {10.14569/IJACSA.2021.0120845}
}

@inproceedings{devlin2019bert,
  author    = {Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
  title     = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year      = {2019},
  pages     = {4171--4186},
  doi       = {10.18653/v1/N19-1423}
}

@inproceedings{kim2014convolutional,
  author    = {Yoon Kim},
  title     = {Convolutional Neural Networks for Sentence Classification},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2014},
  pages     = {1746--1751},
  doi       = {10.3115/v1/D14-1181}
}

@article{hochreiter1997long,
  author  = {Sepp Hochreiter and Jürgen Schmidhuber},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  year    = {1997},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  doi     = {10.1162/neco.1997.9.8.1735}
}

@article{roy2020machine,
  author  = {Partha Kundu Roy and Sushovan Saha Chowdhary and Rashi Bhatia},
  title   = {A Machine Learning Approach for Automation of Resume Recommendation System},
  journal = {Procedia Computer Science},
  year    = {2020},
  volume  = {167},
  pages   = {2318--2327},
  doi     = {10.1016/j.procs.2020.03.284}
}

@inproceedings{ramos2003using,
  author    = {Juan Ramos},
  title     = {Using TF-IDF to Determine Word Relevance in Document Queries},
  booktitle = {Proceedings of the First Instructional Conference on Machine Learning},
  year      = {2003},
  pages     = {133--142}
}

@misc{sanh2019distilbert,
  author       = {Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  title        = {DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter},
  year         = {2019},
  howpublished = {arXiv preprint arXiv:1910.01108},
  doi          = {10.48550/arXiv.1910.01108}
}

@inproceedings{zhang2017sensitivity,
  author    = {Ye Zhang and Byron Wallace},
  title     = {A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification},
  booktitle = {Proceedings of the Eighth International Joint Conference on Natural Language Processing},
  year      = {2017},
  pages     = {253--263}
}

@inproceedings{zhou2016attention,
  author    = {Peng Zhou and Wei Shi and Jun Tian and Zhenyu Qi and Bingchen Li and Hongwei Hao and Bo Xu},
  title     = {Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  year      = {2016},
  pages     = {207--212},
  doi       = {10.18653/v1/P16-2034}
}

@inproceedings{vaswani2017attention,
  author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Łukasz Kaiser and Illia Polosukhin},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  volume    = {30},
  pages     = {5998--6008}
}

@inproceedings{micikevicius2018mixed,
  author    = {Paulius Micikevicius and Sharan Narang and Jonah Alben and Gregory Diamos and Erich Elsen and David Garcia and Boris Ginsburg and Michael Houston and Oleksii Kuchaiev and Ganesh Venkatesh and Hao Wu},
  title     = {Mixed Precision Training},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2018},
  note      = {arXiv:1710.03740}
}

@misc{fastapi2018,
  author       = {Sebastián Ramírez},
  title        = {FastAPI Framework, High Performance, Easy to Learn, Fast to Code, Ready for Production},
  year         = {2018},
  howpublished = {\url{https://fastapi.tiangolo.com/}},
  note         = {Accessed: Oct. 30, 2025}
}

@article{merkel2014docker,
  author  = {Dirk Merkel},
  title   = {Docker: Lightweight Linux Containers for Consistent Development and Deployment},
  journal = {Linux Journal},
  year    = {2014},
  volume  = {2014},
  number  = {239},
  pages   = {Article 2}
}
```
