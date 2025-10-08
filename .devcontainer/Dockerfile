FROM rocm/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && usermod -aG video vscode

# Install Python packages for LLM development
RUN pip install --no-cache-dir \
    tiktoken \
    matplotlib \
    seaborn \
    tqdm \
    jupyter \
    jupyterlab \
    ipywidgets \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard \
    black \
    isort \
    flake8 \
    pytest

# Set up working directory
WORKDIR /workspaces/llm-scratch

# Switch to vscode user
USER vscode

# Set environment variables for ROCm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV HIP_VISIBLE_DEVICES=0