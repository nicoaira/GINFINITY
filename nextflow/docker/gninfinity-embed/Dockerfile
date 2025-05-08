FROM python:3.9-slim

# set working dir
WORKDIR /app

# make sure apt deps for torch‑geometric & networkx
RUN apt-get update && apt-get install -y build-essential cmake && rm -rf /var/lib/apt/lists/*

# copy embedding script and your src/ tree
COPY predict_embedding.py ./predict_embedding.py
COPY src/ ./src/

# install python deps
RUN pip install --no-cache-dir \
      torch \
      torch-geometric \
      pandas \
      tqdm \
      networkx \
      forgi

ENTRYPOINT ["python3","predict_embedding.py"]