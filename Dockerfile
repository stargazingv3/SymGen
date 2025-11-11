FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by evaluation scripts
# divide_function_name.py uses averaged_perceptron_tagger and wordnet
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)"

COPY . .

# This allows you to pip install packages on-the-fly inside the
# running container as a non-root user without permission errors.

RUN mkdir -p /app/results /app/.cache /app/.local /app/.config && \
    chmod -R 777 /app/results /app/.cache /app/.local /app/.config

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

#CMD ["bash"]
CMD ["tail", "-f", "/dev/null"]
