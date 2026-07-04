FROM python:3.11-slim

# install ffmpeg for Whisper audio transcription
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_railway.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY project_youtube_qa.py ./app.py
COPY static ./static

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
