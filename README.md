Voice Authenticator Web Application
![alt text](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

![alt text](https://img.shields.io/badge/Flask-2.x-green.svg)

![alt text](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-orange)

![alt text](https://img.shields.io/badge/Docker-Ready-blue.svg)

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
A modern, web-based voice authentication system that allows users to sign up and log in using only their voice. This project leverages a state-of-the-art deep learning model for speaker verification, providing a robust and accurate biometric security layer. It is built with a Flask backend, a vanilla JavaScript frontend, and is ready for containerized deployment with Docker.
<!-- Add a screenshot or GIF of the application in action here -->
<!-- ![App Screenshot](path/to/your/screenshot.png) -->
Key Features
Intuitive Web Interface: A clean and simple UI for user registration and authentication.
Secure User Registration: Create a new user profile with a unique username.
State-of-the-Art Voice Enrollment: Record a voice passphrase directly in the browser to create a highly discriminative "speaker embedding."
Reliable Voice Authentication: Log in by speaking the same passphrase, with verification powered by a deep learning model.
Real-time Audio Handling: Uses the browser's Web Audio API to capture and process audio seamlessly.
Powerful Backend: A Flask-based server processes audio, generates embeddings, and manages user profiles.
Persistent Storage: User data and voice profiles are stored in an SQLite database.
Dockerized for Easy Deployment: Includes a Dockerfile for hassle-free setup and deployment.
Technology Stack
**Backend: Python, Flask
Frontend: HTML5, CSS3, JavaScript (ES6 Modules)
Voice Biometrics: SpeechBrain, Hugging Face Transformers (speechbrain/spkrec-ecapa-voxceleb)
Database: SQLite
Deployment: Docker**
How It Works
The application follows a client-server architecture to provide a seamless user experience while handling complex biometric processing on the backend.
**Frontend (Client-Side):** The user interacts with the index.html page. All client-side logic is handled by modular JavaScript files. audio.js uses the Web Audio API to record the user's voice, which is then sent as a blob to the backend via API calls defined in api.js.
**Backend (Server-Side):** The Flask application (app.py) receives the audio data. Instead of using traditional methods, it leverages a powerful pre-trained model to create a speaker embedding.
**Advanced Voice Biometrics:** SpeechBrain ECAPA-TDNN
To achieve a high level of accuracy and robustness, this project uses speechbrain/spkrec-ecapa-voxceleb, a state-of-the-art model for speaker verification. This model is based on the ECAPA-TDNN (Emphasized Channel Attention, Propagation, and Aggregation in Time Delay Neural Network) architecture.
**Why this model is used:**
**Superior Accuracy:** The model was trained on the massive VoxCeleb dataset. It achieves an Equal Error Rate (EER) of 0.95% on the VoxCeleb1 test set, signifying extremely high accuracy. A lower EER is better.
**Robustness:** Training on diverse data makes the model resistant to variations in accents, background noise, and recording quality.
**Powerful Speaker Embeddings:** The model converts any voice input into a compact vector (an "embedding") that uniquely represents the speaker's vocal characteristics. Comparing these embeddings is far more reliable than comparing raw audio features.
**Integration into the workflow:**
**Enrollment:** When a user registers, their recorded audio is fed into the ECAPA-TDNN model. The model outputs a unique speaker embedding vector, which is stored in the speaker_voice_profiles_v3.db database.
**Authentication:** During a login attempt, a new embedding is generated from the live audio. The backend then calculates the Cosine Similarity between this new embedding and the stored one. If the score is above a confidence threshold, access is granted.
Project Structure
Generated code
.
├── app.py                   # Main Flask application logic and API endpoints
├── config.py                # Configuration settings for the app
├── Dockerfile               # Instructions to build the Docker image
├── .dockerignore            # Files to exclude from the Docker image
├── requirements.txt         # Python dependencies
├── speaker_voice_profiles_v3.db # SQLite database for user profiles
│
├── static/                  # Static assets (CSS, JS)
│   ├── css/style.css
│   └── js/
│       ├── modules/         # Modular JavaScript for different functionalities
│       │   ├── api.js       # Handles communication with the backend API
│       │   ├── audio.js     # Manages microphone access and audio recording
│       │   ├── auth.js      # Logic for the authentication flow
│       │   ├── config.js    # Frontend configuration
│       │   ├── enrollment.js# Logic for the enrollment flow
│       │   └── ui.js        # Handles UI updates and user feedback
│       └── app.js           # Main frontend script
│
├── templates/               # HTML templates
│   └── index.html           # Single-page application layout
│
└── venv/                    # Virtual environment directory (ignored by git)
Use code with caution.
Installation and Setup
Method 1: Local Development
Prerequisites:
Python 3.8+
pip and venv
Steps:
Clone the repository:
Generated bash
git clone https://github.com/driizzyybreezzyy/Voice-Authenticator.git
cd Voice-Authenticator
Use code with caution.
Bash
Create and activate a virtual environment:
Generated bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
Use code with caution.
Bash
Install the required Python packages:
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
Run the Flask application:
Generated bash
python app.py
Use code with caution.
Bash
Open your web browser and navigate to http://127.0.0.1:5000.
Method 2: Using Docker
Prerequisites:
Docker installed and running.
Steps:
Build the Docker image:
Generated bash
docker build -t voice-authenticator .
Use code with caution.
Bash
Run the Docker container:
Generated bash
# This command maps port 5000 of the container to port 5000 on your machine.
docker run -p 5000:5000 voice-authenticator
Use code with caution.
Bash
Open your web browser and navigate to http://127.0.0.1:5000.
How to Use the Application
Register a New User:
Navigate to the registration section on the web page.
Enter your desired username.
Click the "Record" button and say the required passphrase when prompted.
Your unique voice embedding will be securely saved.
Log In:
Navigate to the login section.
Enter the username you registered with.
Click "Record" and say the same passphrase again.
The system will compare your live voice to the stored profile and grant or deny access based on the similarity score.
Disclaimer
This project is a functional proof-of-concept intended for educational and demonstrative purposes. While it uses a state-of-the-art model, deploying any biometric system in a real-world, high-security production environment requires further security hardening, comprehensive testing, and adherence to privacy regulations.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
