# 🏡 Real Estate Investment Advisor (ML + Streamlit)

A full end-to-end Machine Learning system that predicts:
- Whether a property is a **Good Investment**
- The **future price of a property after 5 years**

This project includes **EDA, preprocessing, feature engineering, baseline modeling, hyperparameter tuning**, and a **Streamlit web app** — all built using a professional modular ML pipeline.

---

## 🚀 Features
- Real estate domain-specific feature engineering
- Handles **classification + regression** together
- Tuned Random Forest models stored for production
- Proper `src/` pipeline like real ML startups
- Modern Streamlit UI for instant predictions
- Reproducible experiments via notebooks + pipelines

---

## 📂 Project structure
```bash
REAL_ESTATE_INVESTMENT_ADVISOR/

├── README.md                        # Project overview & quickstart
├── LICENSE                          # Project license
├── pyproject.toml / setup.cfg       # Optional packaging / dev tools
├── requirements.txt                 # Primary dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── .gitignore
├── .github/                         # CI workflows (tests, lint, notebooks)
│   └── workflows/ci.yml
├── data/
│   ├── raw/                         # Small sample raw CSVs (do NOT commit large datasets)
│   └── processed/                   # Canonical processed snapshot used for demos/tests (small)
├── docs/                            # Architecture & reproducibility docs
├── notebooks/                       # Cleaned notebooks (outputs stripped)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_hyperparameter_tuning.ipynb
├── scripts/                         # Utility scripts for data generation & validation
│   ├── generate_processed.py
│   └── validate_processed.py
├── src/                             # Production-ready package (importable)
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_config.json
│   ├── models/
│   │   ├── train.py
│   │   ├── tuning.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── mlflow_setup.py
│   ├── app/                         # App entrypoints (Streamlit / example scripts)
│   │   └── streamlit_app.py
│   └── api/                         # Optional: FastAPI serving code
├── models/                          # Optional: tiny sample model + metadata (do NOT commit large artifacts)
│   └── metadata.json                # Model provenance and pointers (required if models present)
└── tests/                           # Unit & integration tests (pytest)
    ├── test_features.py
    └── test_train_save.py
```

Run the Streamlit demo locally:
```bash
# from repo root
streamlit run src/app/streamlit_app.py
```

Notes:
- Do not commit large model artifacts, experiment runs, or raw datasets to the repo; use MLflow, S3 or a dedicated artifact store instead.
- Keep notebooks as demonstrations only; move shared logic into `src/` to make code production-ready.

---

## 🔁 ML Pipeline Workflow
1️⃣ Load dataset & perform EDA  
2️⃣ Automated feature engineering  
3️⃣ Preprocessing & train/test split  
4️⃣ Baseline ML models  
5️⃣ Hyperparameter tuning  
6️⃣ Save final models → `/models/`  
7️⃣ Deploy with Streamlit 🚀

---

## 📊 Model Results
| Task | Best Model | Metric | Score |
|------|-----------|--------|------|
| Investment Classification | Random Forest | F1 Score | ⭐ 1.00 |
| Future Price Regression | Random Forest | R² Score | ⭐ 0.84 |

> Scores are high because the dataset is synthetic & rule-based.

---

## 🎯 Streamlit App
✔ Predicts:
- Future Property Price (Lakhs)
- Good vs Bad Investment
- Investment Score Breakdown

Run locally:
```bash
streamlit run Streamlit_app.py
```

---

## ▶️ Installation & Setup
```bash
# Create and activate environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run Streamlit_app.py
```

---

## 🛠 Tech Stack
| Layer | Tools |
|------|------|
| ML & Preprocessing | Scikit-learn |
| App UI | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Code Architecture | Modular `src/` package |

---

## 🔮 Future Enhancements
- Replace synthetic dataset with real housing market data  
- Add SHAP explainability for investment decisions  
- Deploy app to Streamlit Cloud for global access  
- ROI calculator & investment risk scoring  
- CNN model to evaluate property images  

---

## 👤 Author
**Akshay**  
Data Analyst & ML Engineer in progress  
India 🇮🇳

