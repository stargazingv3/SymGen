FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    gnupg \
    expect \
    && rm -rf /var/lib/apt/lists/*

# Install OpenJDK 21 using Eclipse Temurin (matching Ghidra's official Docker setup)
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends temurin-21-jdk && \
    rm -rf /var/lib/apt/lists/*

# Create a user entry to avoid getpass.getuser() errors when running as non-root
# This allows PyTorch to work properly when container runs with user: "${UID}:${GID}"
RUN useradd -m -u 1001 -s /bin/bash appuser || true

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by evaluation scripts
# divide_function_name.py uses averaged_perceptron_tagger and wordnet
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)"

# Install Ghidra (in /opt to avoid being overwritten by volume mounts)
ENV GHIDRA_VERSION=11.4.2
ENV GHIDRA_INSTALL_DIR=/opt/ghidra
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
# Set LD_LIBRARY_PATH to include CUDA libraries (for PyTorch GPU support), system libs, and Java libraries (for Ghidra)
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/nccl/lib:/usr/lib/jvm/temurin-21-jdk-amd64/lib/:/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/

RUN mkdir -p /opt/ghidra && \
    wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_20250826.zip -O /tmp/ghidra.zip && \
    unzip -q /tmp/ghidra.zip -d /tmp && \
    mv /tmp/ghidra_${GHIDRA_VERSION}_PUBLIC/* /opt/ghidra/ && \
    chmod +x /opt/ghidra/support/analyzeHeadless && \
    chmod +x /opt/ghidra/support/launch.sh && \
    rm -rf /tmp/ghidra.zip /tmp/ghidra_${GHIDRA_VERSION}_PUBLIC && \
    echo "Ghidra installed successfully"

# Configure Ghidra's launch.properties with Java 21 path
RUN sed -i 's|^JAVA_HOME_OVERRIDE=$|JAVA_HOME_OVERRIDE=/usr/lib/jvm/temurin-21-jdk-amd64|' /opt/ghidra/support/launch.properties || \
    echo "JAVA_HOME_OVERRIDE=/usr/lib/jvm/temurin-21-jdk-amd64" >> /opt/ghidra/support/launch.properties

# Create a wrapper script for analyzeHeadless that sets JAVA_HOME and LD_LIBRARY_PATH
RUN echo '#!/bin/bash\n\
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64\n\
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/nccl/lib:/usr/lib/jvm/temurin-21-jdk-amd64/lib/:/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/\n\
exec /opt/ghidra/support/analyzeHeadless "$@"' > /usr/local/bin/analyzeHeadless && \
    chmod +x /usr/local/bin/analyzeHeadless

# Add Ghidra to PATH
ENV PATH="${PATH}:/opt/ghidra/support:/usr/local/bin"

COPY . .

# This allows you to pip install packages on-the-fly inside the
# running container as a non-root user without permission errors.

RUN mkdir -p /app/results /app/.cache /app/.local /app/.config /app/.cache/torch /app/ghidra_projects && \
    chmod -R 777 /app/results /app/.cache /app/.local /app/.config /app/ghidra_projects

ENV PYTHONUNBUFFERED=1

#CMD ["bash"]
CMD ["tail", "-f", "/dev/null"]
