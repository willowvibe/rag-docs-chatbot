FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ragchat ./ragchat
COPY web ./web
COPY corpus ./corpus
COPY eval ./eval

EXPOSE 8000
CMD ["uvicorn", "ragchat.api:app", "--host", "0.0.0.0", "--port", "8000"]
