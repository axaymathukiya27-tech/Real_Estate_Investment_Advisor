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

## 📂 Folder Structure
```bash
Real_Estate_Projects/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_classification_model.pkl
│   └── best_regression_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_model_tuning.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   └── models/
│
├── Streamlit_app.py
├── requirements.txt
└── README.md
```

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
.\venv\Scripts\activate    # Windows

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
