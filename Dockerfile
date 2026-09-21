FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860"]