FROM python:3.12-slim

# (optional) system deps if needed for numpy/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app (includes model.joblib, app.py, predict.py, model_meta.json)
COPY app ./app

EXPOSE 8050
# IMPORTANT: Dash 3 uses app.run(...), ensure app.py has: app.run(host="0.0.0.0", port=8050, debug=False)
CMD ["python", "app/app.py"]