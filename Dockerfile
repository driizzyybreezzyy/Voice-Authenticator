# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# --- FIX ---
# Install build-essential (which includes gcc) BEFORE running pip install
# This is needed to compile packages like webrtcvad
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the pip cache, making the image smaller
# --compile: Compiles .py files to .pyc for potentially faster startup
RUN pip install --no-cache-dir --compile -r requirements.txt

# Copy the rest of the application code into the container at /app
# (Make sure .dockerignore is set up to exclude venv, .git, etc.)
COPY . .

# Make port 5000 available to the world outside this container
# This is the port Gunicorn will listen on INSIDE the container.
EXPOSE 5000

# Define environment variables (optional, but good practice)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# For Gunicorn, you might also set PYTHONUNBUFFERED=1 for better logging

# Command to run the application using Gunicorn
# Gunicorn will look for an app instance named 'app' in the file 'app.py'
# -w 4: Number of worker processes (adjust based on your app and server resources)
# --bind 0.0.0.0:5000: Listen on all network interfaces on port 5000
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]
