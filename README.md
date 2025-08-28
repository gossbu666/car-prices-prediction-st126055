🚗 Car Price Prediction (Machine Learning Assignment)

📌 Project Overview

This project develops a car price prediction system.
The solution includes:
	•	Task 1: Dataset Preparation & Model Training (EDA, preprocessing, model selection, inference).
	•	Task 2: Report (summary & analysis of results).
	•	Task 3: Deployment (Dash web app for price prediction).

Dataset used: Cars dataset

⸻

📂 Repository Structure

.
├── app/                     # Web application folder
│   ├── app.py               # Dash app code
│   ├── Dockerfile           # Dockerfile for app
│   ├── docker-compose.yml   # Compose file
│   └── model.joblib         # Trained model (⚠️ excluded, >100MB)
├── data/Cars.csv            # Raw dataset
├── notebooks/st126055_CarPrice.ipynb  # Jupyter notebook (EDA, training, evaluation)
├── A1__Predicting_Car_Price.pdf       # Assignment instructions
├── README.md                # This file
└── requirements.txt         # Python dependencies

⚠️ Note on model.joblib
The trained model is not included in this repo due to GitHub’s 100MB limit.
➡️ To reproduce: run the notebook st126055_CarPrice.ipynb, which will save the trained model into app/model.joblib.

🛠️ Setup & Usage

1. Install dependencies

pip install -r requirements.txt

2. Run Notebook (Training + Export Model)

jupyter notebook notebooks/st126055_CarPrice.ipynb

This will train the model and save it as:

app/model.joblib

3. Run Dash app (locally)

cd app
python app.py

Then open: 👉 http://localhost:8050

4. Run with Docker

docker compose up --build

📝 Notes & Issues
	•	The Random Forest model was chosen as the final model because it achieved the best performance (MAE ≈ 28K, RMSE ≈ 126K, R² ≈ 0.98).
	•	One challenge was that the trained model file exceeded GitHub’s 100MB limit.
	•	✅ Solution: we compressed the model into a smaller version, which still works properly for inference in the Dash app.
	•	If you want maximum precision, you can stop at the original (uncompressed) model step in the notebook and run Dash directly with that version.

⸻

👤 Author
	•	Supanut Kompayak (st126055)
	•	Course: AT82.03 Machine Learning
	•	Asian Institute of Technology (AIT)
