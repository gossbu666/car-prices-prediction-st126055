FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app ./app


EXPOSE 8050
CMD ["python", "-u", "app/app.py"]