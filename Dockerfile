FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying app code so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the artifacts the inference service reads at runtime
COPY models/ ./models/
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
