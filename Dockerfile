FROM python:3.11-slim

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY notebooks/requirements_quantum.txt .

RUN pip install --no-cache-dir -r requirements_quantum.txt

COPY quantum/ ./quantum/

COPY notebooks/OptimizationProblemData.json ./quantum/

WORKDIR /app/quantum

CMD ["python", "fully_quantum_approach.py"]
