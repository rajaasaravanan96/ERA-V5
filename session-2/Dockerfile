FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["gunicorn", "app:app", "--worker-class", "gthread", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:5000"]
