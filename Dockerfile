FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
# Pass the key at runtime: docker run -p 8501:8501 --env-file .env job-match-agent
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
