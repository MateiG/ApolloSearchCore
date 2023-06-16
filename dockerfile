# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV OPENAI_API_KEY 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
ENV GOOGLE_APPLICATION_CREDENTIALS /app/credentials/apollosearch-bebb5b92d946.json

# Run app.py when the container launches
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
