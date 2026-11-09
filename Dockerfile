# CPU inference image for the neurodx API.
#   docker build -t neurodx .
#   docker run -p 8000:8000 -v "$PWD/models:/models:ro" neurodx    # expects /models/best.pt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY neurodx ./neurodx
RUN useradd --create-home --uid 10001 app
USER app

ENV NEURODX_CHECKPOINT=/models/best.pt \
    NEURODX_ABSTAIN_BELOW=0.6 \
    MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "neurodx.serve:app", "--host", "0.0.0.0", "--port", "8000"]
