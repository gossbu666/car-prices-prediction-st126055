**Car Price Prediction (Machine Learning Assignment)**

**Project Overview**

This project develops a car price prediction system.
The solution includes:
	•	Task 1: Dataset Preparation & Model Training (EDA, preprocessing, model selection, inference).
	•	Task 2: Report (summary & analysis of results).
	•	Task 3: Deployment (Dash web app for price prediction).

Dataset used: Cars dataset

⸻

## 1) Task 1 — Data Prep & Modeling

Steps covered in the notebook:
- Cleaning rules (from assignment):
  - Map **owner**: `{"First Owner":1, "Second Owner":2, "Third Owner":3, "Fourth & Above Owner":4, "Test Drive Car":5}`
  - Drop rows where **fuel** ∈ {CNG, LPG}
  - Convert **mileage** (remove `kmpl`), **engine** (remove `CC`), **max_power** (remove unit) → numeric
  - Extract **brand** = first word of `name`
  - Drop **torque**
  - Remove **Test Drive Car** records
- Target transform: `y = np.log(selling_price)`; inverse at inference: `np.exp(pred)`
- Preprocessing:
  - Categorical: impute mode + OneHotEncoder(ignore_unknown)
  - Numeric: engine/max_power → median; mileage → mean; seats → mode; pass-through year/km_driven/owner
- Model comparison: Linear, SVR, KNN, Decision Tree, Random Forest
- Hyperparameter tuning (GridSearchCV) for RandomForest
- Feature importance (raw + grouped by original feature)
- Inference example

**Best model**: RandomForest (post GridSearch) with strong generalization (R² ≈ 0.98 on test).

---

## 2) Task 2 — Report

See the last section of `notebooks/st126055_CarPrice.ipynb` (2–3 paragraphs):  
- Which features are important and why  
- Which algorithms work well for this tabular problem and why RandomForest performed best  

---

## 3) Task 3 — Web App (Dash)


### Run locally (Python)
```bash
# from project root
pip install -r requirements.txt
python app/app.py
# open http://127.0.0.1:8050
--

**Setup & Usage**
- Run Dash app (locally)

cd app
python app.py

Then open: 👉 http://localhost:8050

- Run with Docker

docker compose up --build


**Notes & Issues**
	The Random Forest model was chosen as the final model because it achieved the best performance (MAE ≈ 28K, RMSE ≈ 126K, R² ≈ 0.98).
⸻

**👤 Author**
	•	Supanut Kompayak (st126055)
	•	Course: AT82.03 Machine Learning
	•	Asian Institute of Technology (AIT)
