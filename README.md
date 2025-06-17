Voice Authenticator Web App

![alt text](https://img.shields.io/badge/Python-3.8%2B-blue.svg)


![alt text](https://img.shields.io/badge/Flask-2.x-green.svg)


![alt text](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-orange)


![alt text](https://img.shields.io/badge/Docker-Ready-blue.svg)


![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

A web application for user authentication using voice biometrics. Built with Flask, JavaScript, and a state-of-the-art SpeechBrain model for high accuracy.

Core Features

Web-based user registration and login.

Voice enrollment to create a unique speaker profile (embedding).

Authentication powered by the speechbrain/spkrec-ecapa-voxceleb model.

Simple UI with real-time audio capture from the browser.

Dockerized for easy and consistent deployment.

How It Works

Enrollment: A user's voice is recorded and converted into a unique "speaker embedding" using the SpeechBrain model. This embedding is saved to the database.

Authentication: A new voice recording is converted into a new embedding.

Verification: The new embedding is compared to the saved one using Cosine Similarity. A similarity score above a set threshold grants access.

Tech Stack

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript

Voice Model: SpeechBrain / Hugging Face Transformers

Database: SQLite

Deployment: Docker

Quickstart
Method 1: Local Setup

Clone the repository:

Generated bash
git clone https://github.com/driizzyybreezzyy/Voice-Authenticator.git
cd Voice-Authenticator


Create a virtual environment and install dependencies:

Generated bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Run the application:

Generated bash
python app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Open your browser to http://127.0.0.1:5000.

Method 2: Docker

Build the Docker image:

Generated bash
docker build -t voice-authenticator .
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Run the container:

Generated bash
docker run -p 5000:5000 voice-authenticator
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Open your browser to http://127.0.0.1:5000.

License

Licensed under the MIT License.
