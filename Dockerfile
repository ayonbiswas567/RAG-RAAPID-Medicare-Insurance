# Container image for the HTTP micro-service.
# NOTE: written for completeness and reviewed by hand; it was not built in the development
# environment (the CPU torch wheel is large). Build with:  docker build -t raapid-rag .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY eval ./eval

# CPU-only torch keeps the image far smaller than the default CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

# Bake the embedding model and the prebuilt index into the image so a container starts with no
# network access and no cold embedding step. After this, the runtime can set HF_HUB_OFFLINE=1.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && raapid-rag ingest --json > /dev/null

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=4).raise_for_status()"

# OPENAI_API_KEY is injected at run time, never baked into the image.
CMD ["uvicorn", "raapid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
