FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

COPY . /app/

ENV GOOGLE_APPLICATION_CREDENTIALS credentials/apollosearch-bebb5b92d946.json
ENV OPENAI_API_KEY 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

EXPOSE 5000
CMD ["python", "app.py"]
