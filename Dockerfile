FROM python:3.10-slim

WORKDIR /app

# C/C++ toolchain for the advanced scientific libraries; curl for Ollama init.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir ".[app,advanced]"

EXPOSE 8501

CMD ["proteomics-meta-app", "--server.port=8501", "--server.address=0.0.0.0"]
