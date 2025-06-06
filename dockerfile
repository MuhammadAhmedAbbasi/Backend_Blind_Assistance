FROM python:3.10  # Removed '-slim'

# System dependencies (optional but recommended)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app
EXPOSE 8888

CMD ["uvicorn", "Service.main:app", "--host", "0.0.0.0", "--port", "8888"]