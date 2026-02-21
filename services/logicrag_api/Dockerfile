FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Pull upstream LogicRAG implementation directly from source.
RUN git clone --depth=1 https://github.com/chensyCN/LogicRAG.git /opt/logicrag

RUN pip install --no-cache-dir -r /opt/logicrag/requirements.txt

COPY services/logicrag_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/logicrag_api /app

ENV PYTHONPATH=/opt/logicrag

EXPOSE 8100

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
