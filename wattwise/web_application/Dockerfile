FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY streamlit_app.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
