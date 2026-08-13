![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5.svg)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)
![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)

## System Architecture

```text
                    Job Description
                           │
                           ▼
                 Text Preprocessing
                           │
                           ▼
                   TF-IDF Vectorizer
                           │
                           ▼
Resume ──► Resume Parser ─► Skill Extraction
                           │
                           ▼
                 Cosine Similarity Engine
                           │
                           ▼
                 ATS Compatibility Score
                           │
                           ▼
                 Candidate Ranking Engine
                           │
                           ▼
                  PDF Report Generator
                           │
                           ▼
                 Streamlit Dashboard
```


## AI Workflow

1. Upload Resume
2. Parse Resume (PDF/DOCX/TXT)
3. Clean and preprocess text
4. Extract structured information
5. Extract skills using NLP
6. Analyze Job Description
7. Compute TF-IDF vectors
8. Calculate Cosine Similarity
9. Generate ATS Compatibility Score
10. Rank candidates
11. Generate recruiter report



## System Requirements

- Python 3.10+
- pip
- Windows / Linux / macOS
- 4 GB RAM (Minimum)
- Internet required only for first-time model downloads


## Running the Application

```bash
streamlit run app.py
```

The application will open automatically at:

```
http://localhost:8501
```


## Contributor

- Shivam Kumar



## Future Enhancements

- OCR support for scanned resumes
- AI-generated interview questions
- Resume grammar analysis
- Recruiter authentication
- Candidate database
- Email notifications
- Job recommendation engine
- Resume improvement suggestions
- Multi-language resume support
- Cloud deployment
- REST API integration


## Why HireSense AI?

Recruiters often receive hundreds of resumes for a single job opening. Manual screening is time-consuming and prone to inconsistency.

HireSense AI automates the initial screening process using Natural Language Processing (NLP) and Machine Learning techniques. By extracting relevant information, matching candidate skills with job requirements, and generating an explainable compatibility score, the system helps recruiters identify suitable candidates more efficiently.

The project is designed as an educational demonstration of AI-assisted recruitment rather than a replacement for human decision-making.


## Acknowledgements

This project makes use of the following open-source libraries and frameworks:

- Python
- Streamlit
- Scikit-learn
- spaCy
- NLTK
- Pandas
- NumPy
- ReportLab
- PyPDF2
- pdfplumber
- python-docx


