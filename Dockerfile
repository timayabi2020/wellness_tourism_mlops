FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN python --version && \
    pip --version && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=7860"]