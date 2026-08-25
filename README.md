# 🐦 BirdSense-AI

> **AI-powered bird species identification from sound**

BirdSense-AI is an end-to-end machine learning application that identifies bird species from audio recordings. The system extracts acoustic features from bird calls using **Librosa**, processes them with a **TensorFlow/Keras CNN**, and exposes the trained model through a **FastAPI** backend with a **React + Vite** frontend.

---

## ✨ Features

- 🎵 Upload bird audio recordings
- 🧠 AI-based bird species classification
- 🔊 Supports `.mp3`, `.wav`, `.ogg`, `.flac`, and `.m4a`
- 📊 Displays prediction confidence
- 🏆 Shows top 5 predicted species
- ⚡ FastAPI REST API for model inference
- ⚛️ React + Vite frontend
- 🎯 MFCC-based acoustic feature extraction
- 🔄 End-to-end frontend → API → ML model pipeline
- 🧹 Automatically removes temporary uploaded files

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │    React Frontend   │
                    │      Vite + UI      │
                    └──────────┬──────────┘
                               │
                         Audio Upload
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │      /predict       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Audio Loading    │
                    │       Librosa       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MFCC Extraction   │
                    │  Acoustic Features  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TensorFlow / Keras  │
                    │     CNN Model       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Species + Confidence│
                    │     Top 5 Results   │
                    └─────────────────────┘
🛠️ Tech Stack
Machine Learning
Python
TensorFlow / Keras
NumPy
Librosa
Scikit-learn
MFCC acoustic features
Backend
FastAPI
Uvicorn
Python
REST API
CORS
Frontend
React
Vite
JavaScript
CSS
Tools
Git & GitHub
VS Code
Python Virtual Environment
📁 Project Structure
BirdSense-AI/
│
├── backend/
│   ├── dataset/
│   │
│   ├── features/
│   │   ├── dataset.npz
│   │   ├── labels.json
│   │   └── failed_files.json
│   │
│   ├── models/
│   │   └── bird_classifier.keras
│   │
│   ├── saved_models/
│   ├── uploads/
│   │
│   ├── app.py
│   ├── config.py
│   ├── extract_features.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── scan_dataset.py
│   ├── train.py
│   └── utils.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
🚀 Getting Started
1. Clone the Repository
git clone https://github.com/anjalisaini89/BirdSense-AI.git
cd BirdSense-AI
2. Create Python Virtual Environment

On Windows PowerShell:

python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1

Install backend dependencies:

cd backend
pip install -r ../requirements.txt
🧠 Run the Backend

From the backend directory:

uvicorn app:app --reload

The API will run at:

http://127.0.0.1:8000

FastAPI Swagger documentation:

http://127.0.0.1:8000/docs
❤️ Health Check

Endpoint:

GET /health

Example response:

{
  "status": "healthy"
}
🎵 Bird Prediction API

The main endpoint is:

POST /predict

It accepts an audio file using multipart/form-data.

Supported formats
.wav
.mp3
.ogg
.flac
.m4a
Using Swagger
Open:
http://127.0.0.1:8000/docs
Open POST /predict
Click Try it out
Select an audio file
Click Execute

Example response:

{
  "success": true,
  "filename": "Andean Guan10.mp3",
  "prediction": {
    "species": "Andean Tinamou_sound",
    "confidence": 37.35,
    "top_predictions": [
      {
        "species": "Andean Tinamou_sound",
        "confidence": 37.35
      },
      {
        "species": "Darwins Nothura_sound",
        "confidence": 10.54
      },
      {
        "species": "Cauca Guan_sound",
        "confidence": 6.04
      },
      {
        "species": "Blue-throated Piping Guan_sound",
        "confidence": 4.24
      },
      {
        "species": "Hooded Tinamou_sound",
        "confidence": 3.83
      }
    ]
  }
}

Note: Confidence represents the model's predicted probability for the selected class. It should not be interpreted as guaranteed real-world identification accuracy.

⚛️ Run the Frontend

Open a second terminal.

From the project root:

cd frontend
npm install
npm run dev

The frontend will normally run at:

http://localhost:5173

Make sure the FastAPI backend is running simultaneously.

🔬 How the Model Works

BirdSense-AI uses an acoustic classification pipeline.

1. Audio Loading

The uploaded audio is loaded using Librosa and resampled to the configured sample rate.

2. Audio Preprocessing

Audio is trimmed or padded to the configured duration so the model receives a consistent input size.

3. MFCC Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from the audio.

MFCCs provide a compact representation of important characteristics of an audio signal and are commonly used for audio classification.

4. CNN Classification

The MFCC representation is passed into a trained TensorFlow/Keras CNN.

The model generates probabilities for the available bird species.

5. Prediction

The species with the highest probability is selected.

The application also returns the top 5 predictions and their confidence values.

🧪 Command-Line Prediction

The trained model can also be tested directly without the frontend or API.

From backend/:

python predict.py "dataset\Voice of Birds\Voice of Birds\Andean Guan_sound\Andean Guan10.mp3"

Example:

============================================================
BirdSense-AI Prediction
============================================================

Bird Species : Andean Tinamou_sound
Confidence   : 37.35%

Top 5 Predictions:
------------------------------------------------------------
Andean Tinamou_sound                     37.35%
Darwins Nothura_sound                    10.54%
Cauca Guan_sound                           6.04%
Blue-throated Piping Guan_sound            4.24%
Hooded Tinamou_sound                        3.83%
============================================================
📊 Current Status
✅ Completed
 Bird audio dataset setup
 Audio preprocessing
 MFCC feature extraction
 CNN model training
 Saved TensorFlow/Keras model
 Command-line prediction
 FastAPI backend
 /health endpoint
 /predict endpoint
 Audio upload handling
 CORS configuration
 React + Vite frontend
 Frontend/backend integration
 Prediction result UI
 Top 5 prediction display
 Confidence visualization
🚧 Roadmap
 Clean bird species names in the UI
 Improve model accuracy
 Improve confidence calibration
 Add confusion matrix
 Add detailed model evaluation
 Add audio waveform visualization
 Add spectrogram visualization
 Add real-time microphone recording
 Add analysis/loading animation
 Add bird species information
 Add scientific names
 Add habitat information
 Add bird images
 Deploy frontend and backend
 Improve production error handling
 Expand documentation
🎯 Future Vision

BirdSense-AI is designed to evolve into a complete bird acoustic analysis platform.

🎙️ Record Bird Sound
        ↓
🎵 Audio Processing
        ↓
📈 Waveform + Spectrogram
        ↓
🧠 Deep Learning Inference
        ↓
🐦 Species Identification
        ↓
📊 Acoustic Analysis
        ↓
🌍 Species Information

Future versions may include:

Real-time bird call recognition
Larger and more diverse datasets
Advanced deep learning architectures
Environmental noise filtering
Multiple-bird detection
Interactive acoustic visualizations
Geographic bird information
Bird occurrence analytics
⚠️ Limitations

The current model is a machine learning prototype and may produce incorrect predictions, especially when:

Multiple birds are present in the recording
Background noise is significant
Audio quality is poor
A species is underrepresented in the dataset
The recording differs significantly from the training data

Predictions should therefore be treated as model estimates rather than definitive species identification.

🤝 Contributing

Contributions and suggestions are welcome.

Create a feature branch:

git checkout -b feature/your-feature

After making changes:

git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature

Then open a Pull Request.

📜 License

This project is currently intended as a personal/academic machine learning project.

An appropriate open-source license should be added before publicly distributing the project.

👩‍💻 Author

Anjali Saini

B.Tech Computer Science & Engineering — AI & Data Science

GitHub:
https://github.com/anjalisaini89