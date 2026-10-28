FROM python:3.11-slim

WORKDIR /app

# hdbscan builds native extensions from source -- needs a C compiler
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY data_collection/requirements.txt .

# CPU-only torch wheel -- the default PyPI wheel drags in the full NVIDIA CUDA
# toolkit (several GB) even though this container has no GPU to use it with.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY data_collection/sanity_check_query_volume.py .
COPY data_collection/topic_diversity.py .

# HF_TOKEN must be provided at `docker run` time (-e HF_TOKEN=...);
# never bake a token into the image.
ENV HF_HOME=/app/.cache/huggingface
ENV OUTPUT_DIR=/app/outputs

VOLUME ["/app/outputs"]

CMD ["python", "sanity_check_query_volume.py"]
