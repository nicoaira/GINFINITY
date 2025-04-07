# Imagen base de NVIDIA con CUDA 12.1 y Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Variables de entorno para no interactuar con la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcairo2 \
    pkg-config \
    graphviz \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Usar python3.10 como comando por defecto
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copiar requirements
COPY requirements.txt .

# Actualizar pip y setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch 2.4.1 con CUDA 12.1
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Instalar el resto de paquetes del requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch Geometric y sus dependencias (torch 2.4.1 + CUDA 12.1)
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html && \
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html && \
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html && \
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html && \
    pip install torch-geometric

# Copiar tu código al contenedor
COPY . .

# Comando por defecto
CMD ["python", "main.py"]
