FROM python:3.9-slim

WORKDIR /app

# copy distance script
COPY ../../compute_distances.py ./

# install runtime deps
RUN pip install --no-cache-dir \
      pandas \
      torch \
      tqdm

ENTRYPOINT ["python3","compute_distances.py"]