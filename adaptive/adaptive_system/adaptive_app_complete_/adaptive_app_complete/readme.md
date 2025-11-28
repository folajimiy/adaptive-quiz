# Adaptive Java Tutor — Modular Version

This project is an AI-guided adaptive learning system for Java programming courses.  
It combines:

- Adaptive question sequencing  
- Bloom-level progression  
- Subconcept-based remediation  
- Test vs Practice modes  
- Mastery tracking  
- Student dashboards  
- Teacher analytics dashboards  
- Slide-restricted question generation pipeline  

---

## 📁 Project Structure

adaptive_tutor/  
│  
├── main.py — Streamlit router for Student/Teacher  
├── config.py — global constants  
├── README.md — documentation  
│  
├── core/ — business logic  
│   ├── data_access.py — CSV + logs  
│   ├── models.py — dataclasses  
│   ├── mastery_engine.py — mastery & progression  
│   └── adaptive_engine.py — full adaptive engine  
│  
├── ui/ — user interface  
│   ├── student.py — student dashboard + practice/test  
│   └── teacher.py — teacher analytics  
│  
├── generation/ — question generation pipeline  
│   └── question_generation_v5.py  
│  
├── data/ — question bank & students  
└── logs/ — session logs per student  

---

## 🚀 Running the App



streamlit run main.py


---

## 📘 Student Features

- Adaptive question sequencing  
- Bloom-aware progression & demotion  
- Difficulty tiering (easy/medium/hard)  
- Subconcept remediation  
- Confidence-based logging  
- Bookmark questions  
- Review incorrect questions  
- PDF report card  

---

## 👩‍🏫 Teacher Features

- Class-level accuracy  
- Topic mastery heatmaps  
- Bloom-level analysis  
- Test vs practice analytics  
- Student drill-down  
- Misconception detection  
- CSV exports  

---

## 🧠 Adaptive Engine

The adaptive engine uses:

- Confidence-weighted progression  
- High-conf errors → demotion  
- Mastery completion thresholds  
- Weak-subconcept targeting  
- Difficulty weighting (medium preferred)  
- Fallback question pools  

---

## 📄 Question Generation

Comes from `generation/question_generation_v5.py`  
(using slides, GPT-4.x, Claude 3.5 Sonnet refinement, metadata scoring)

---

## 📤 Logging

All responses are logged as CSV under `logs/student_<id>.csv`.

Includes:

- correctness  
- bloom level  
- confidence  
- response time  
- subconcept  
- reasoning  
