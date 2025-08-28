# 🚗 Car Price Prediction (Machine Learning Assignment)

## 📌 Project Overview
This project develops a **car price prediction system**   
The solution includes:
- **Task 1: Dataset Preparation & Model Training** (EDA, preprocessing, model selection, inference).
- **Task 2: Report** (summary & analysis of results).
- **Task 3: Deployment** (Dash web app for price prediction).

Dataset used: **Cars dataset**

---

## 📂 Repository Structure

.
├── app/                     # Web application folder
│   ├── app.py               # Dash app code
│   ├── Dockerfile           # Dockerfile for app
│   ├── docker-compose.yml   # Compose file
│   └── model.joblib         # Trained model (⚠️ excluded, >100MB)
├── Cars.csv                 # Raw dataset
├── st126055_CarPrice.ipynb  # Jupyter notebook (EDA, training, evaluation)
├── A1__Predicting_Car_Price.pdf  # Assignment instructions
├── README.md                # This file
└── .gitignore

⚠️ **Note on `model.joblib`**  
The trained model is not included in this repo due to GitHub’s 100MB limit.  
➡️ To reproduce: run the notebook `st126055_CarPrice.ipynb`, which will save the trained model into `app/model.joblib`.

---

## 🛠️ Setup & Usage

### 1. Run Notebook (Training + Export Model)
```bash
jupyter notebook st126055_CarPrice.ipynb

/app/model.joblib

cd app
python app.py

docker compose up --build


 Author
	•	Supanut Kompayak (st126055)
	•	Course: AT82.03 Machine Learning
	•	Asian Institute of Technology (AIT)