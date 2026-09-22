<div align="center">

# 🐦 BirdSense-AI

### *AI-Powered Bird Sound Recognition & Acoustic Intelligence*

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Poppins&weight=700&size=28&pause=1000&color=67E8F9&center=true&vCenter=true&width=850&lines=AI-Powered+Bird+Sound+Recognition;114-Species+Deep+Learning+Classifier;MFCC+%2B+CNN+Audio+Classification;Interactive+2D+%26+3D+Acoustic+Visualization;Built+with+TensorFlow+%2B+FastAPI+%2B+React" />
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
<img src="https://img.shields.io/badge/Vite-Frontend-646CFF?style=for-the-badge&logo=vite&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>

</p>

<p align="center">

<img src="https://img.shields.io/badge/Species-114-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Test%20Accuracy-52.90%25-06B6D4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Macro%20F1-0.4695-8B5CF6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model%20Parameters-437K-F59E0B?style=for-the-badge"/>

</p>

---

### 🌌 *Listen. Analyze. Identify.*

**BirdSense-AI transforms bird audio into intelligent species predictions using deep learning, audio signal processing, and interactive acoustic visualization.**

</div>

---

# 🌿 What is BirdSense-AI?

BirdSense-AI is an end-to-end AI application designed to identify bird species from audio recordings.

The system takes an uploaded bird recording, processes the audio using **Librosa**, extracts **MFCC features**, and feeds them into a **CNN-based deep learning classifier trained across 114 bird species**.

The trained model is exposed through a **FastAPI REST API** and connected to a modern **React + Vite frontend**.

The interface presents:

- 🐦 Predicted bird species
- 🎯 Prediction confidence
- 🔝 Top-5 predictions
- 🎧 Audio preview
- 📊 Interactive 2D acoustic visualization
- 🌊 Interactive 3D acoustic surface

---

# ✨ Features

## 🐦 AI Bird Recognition

- 🎤 Upload bird recordings
- 🧠 Deep learning-based classification
- 🐦 Classification across **114 species**
- 🎯 Confidence score
- 🔝 Top-5 predictions
- 🎧 Built-in audio preview

---

## 🎛️ Audio Intelligence

BirdSense-AI processes recordings through an audio feature extraction pipeline:

```text
Audio Recording
       ↓
Audio Loading
       ↓
Resampling → 22,050 Hz
       ↓
5-Second Standardization
       ↓
MFCC Feature Extraction
       ↓
40 × 216 Feature Representation
       ↓
CNN Classifier
       ↓
Bird Species Prediction
🌊 Interactive Acoustic Visualization
BirdSense-AI doesn't stop at prediction.
The application also visualizes the acoustic structure of the recording.
📊 2D Spectrogram
Explore the time-frequency representation of the recording through an interactive heatmap.
🌌 3D Acoustic Surface
Rotate, zoom, and explore the acoustic representation in three dimensions.
              Acoustic Intensity
                     ▲
                     │
                ╭────────╮
             ╭──╯        ╰──╮
          ╭──╯               ╰──╮
       ╭──╯                      ╰──╮
      ╰──────────────────────────────╯
       Time  ──────────────────────►
              Frequency
🎮 Interactive Controls
- 🖱️ Drag → Rotate
- 🔍 Scroll → Zoom
- 🖱️ Hover → Inspect acoustic values
- 🔄 Double-click → Reset view
🧠 Deep Learning Model
The final BirdSense-AI model is V6, developed through multiple controlled experiments.
🏗️ V6 Architecture
Input
40 × 216 MFCC
      │
      ▼
Conv2D 32
Batch Normalization
MaxPooling
Dropout
      │
      ▼
Conv2D 64
Batch Normalization
MaxPooling
Dropout
      │
      ▼
Conv2D 128
Batch Normalization
MaxPooling
Dropout
      │
      ▼
Conv2D 256
Batch Normalization
      │
      ▼
Global Average Pooling
      │
      ▼
Dense 128
Dropout
      │
      ▼
Dense 114
Softmax
⚙️ Model Configuration
Parameter	Value
Input Features	40 × 216
Output Classes	114
Parameters	437,362
Batch Size	32
Learning Rate	0.001
Optimizer	Adam
Loss	Categorical Crossentropy
Maximum Epochs	30
Feature Type	MFCC


📈 Model Performance
🏆 V6 Results
Metric	Result
Test Samples	431
Test Classes	107
Test Accuracy	52.90%
Macro F1	0.4695
Weighted F1	0.5247
Test Loss	2.1171
Best Validation Accuracy	64.35%


⚠️ Performance varies between bird species because the dataset is significantly imbalanced and some species have relatively few recordings.

🧪 From V1 → V6
BirdSense-AI wasn't built around a single model experiment.
Several versions were trained to understand how different approaches affected performance.
Version	Experiment	Test Accuracy
V1	Baseline MFCC CNN	46.53%
V2	Class-weighted CNN	42.23%
V3	Controlled train/validation/test split	43.39%
V4	Log-Mel features	18.10%
V5	Standardized Log-Mel features	0.93%
V6	Deeper MFCC CNN	52.90%


🔬 What the experiments showed
The experiments helped compare:
- MFCC vs Log-Mel representations
- Class weighting
- Data splitting strategies
- CNN depth
- Validation behavior
- Per-species classification performance
V6 became the final model after achieving the strongest held-out test performance among these experiments.
📊 Dataset
The processed dataset contains:
2,161 discovered audio files
        ↓
2,158 successfully processed
        ↓
114 bird species
Dataset Characteristics
- 🎵 Environmental bird recordings
- 🐦 114 species
- 🎧 2,158 successfully extracted feature samples
- ⚖️ Uneven class distribution
- 🔬 40 MFCC coefficients
- ⏱️ 5-second standardized audio
The dataset and generated feature files are intentionally excluded from Git version control.
⚙️ Tech Stack
Category	Technologies
Language	Python 3.11, JavaScript
Deep Learning	TensorFlow / Keras
Audio Processing	Librosa
ML Utilities	NumPy, Scikit-learn
Backend	FastAPI, Uvicorn
Frontend	React, Vite
Visualization	Plotly.js, React Plotly
API	REST
Version Control	Git / GitHub


🏗️ System Architecture
                    ┌─────────────────────┐
                    │     React UI        │
                    │                     │
                    │  Audio Upload       │
                    │  Prediction UI      │
                    │  2D Visualization   │
                    │  3D Visualization   │
                    └──────────┬──────────┘
                               │
                               │ REST API
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │                     │
                    │  File Validation    │
                    │  Upload Handling    │
                    │  Prediction Route   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Audio Pipeline    │
                    │                     │
                    │  Librosa            │
                    │  Resampling         │
                    │  MFCC Extraction    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      CNN V6         │
                    │                     │
                    │  114-Class Softmax  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Results  │
                    │                     │
                    │ Species             │
                    │ Confidence          │
                    │ Top-5               │
                    │ Visualization       │
                    └─────────────────────┘
📂 Project Structure
BirdSense-AI/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── utils.py
│   │
│   ├── train.py
│   ├── train_v2.py
│   ├── train_v3.py
│   ├── train_v4.py
│   ├── train_v5.py
│   └── train_v6.py
│   │
│   ├── extract_features.py
│   ├── extract_features_v4.py
│   ├── scan_dataset.py
│   ├── diagnose_features.py
│   └── debug_audio.py
│   │
│   ├── evaluate_model.py
│   ├── evaluate_v2.py
│   ├── evaluate_v3.py
│   └── evaluate_v6.py
│   │
│   └── models/
│       ├── evaluation_v2.txt
│       ├── evaluation_v3.txt
│       ├── evaluation_v6.txt
│       └── training_history*.json
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   ├── birdInfo.js
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── .gitignore
├── README.md
└── requirements.txt
🚀 Getting Started
1. Clone the repository
git clone https://github.com/anjalisaini89/BirdSense-AI.git
2. Navigate into the project
cd BirdSense-AI
3. Create the Python environment
python -m venv venv
4. Activate the environment
Windows PowerShell
.\venv\Scripts\Activate.ps1
5. Install Python dependencies
pip install -r requirements.txt
6. Start the FastAPI backend
cd backend
..\venv\Scripts\Activate.ps1
uvicorn app:app --reload
Backend:
http://localhost:8000
7. Start the frontend
Open another terminal:
cd frontend
npm install
npm run dev
Frontend:
http://localhost:5173
🔌 API Endpoints
Health Check
GET /health
Response:
{
  "status": "healthy"
}
Bird Prediction
POST /predict
Accepts an audio file and returns the predicted bird species, confidence information, top predictions, and visualization data used by the frontend.
Supported formats:
.wav
.mp3
.ogg
.flac
.m4a
🔐 File Handling
Uploaded recordings are temporarily stored using generated UUID filenames.
After prediction completes, the temporary audio file is automatically deleted.
Upload
  ↓
Temporary File
  ↓
Prediction
  ↓
Response
  ↓
File Deleted
🎯 Current Progress
🧠 Machine Learning
- ✅ Dataset scanning
- ✅ Audio preprocessing
- ✅ MFCC extraction
- ✅ Baseline CNN
- ✅ Class-weighting experiment
- ✅ Controlled data split
- ✅ Log-Mel experiment
- ✅ Standardized Log-Mel experiment
- ✅ Deeper CNN architecture
- ✅ V6 model training
- ✅ Held-out evaluation
- ✅ Per-species classification report
⚡ Backend
- ✅ FastAPI application
- ✅ CORS configuration
- ✅ Audio upload endpoint
- ✅ File validation
- ✅ Temporary file handling
- ✅ Model inference
- ✅ Health endpoint
🎨 Frontend
- ✅ React + Vite interface
- ✅ Audio upload
- ✅ Audio preview
- ✅ Prediction results
- ✅ Confidence display
- ✅ Top-5 predictions
- ✅ Interactive 2D visualization
- ✅ Interactive 3D visualization
- ✅ Responsive interface
🗺️ Future Roadmap
- [ ] Continuous microphone inference
- [ ] Audio augmentation
- [ ] Larger and more balanced dataset
- [ ] Transfer learning with pretrained audio models
- [ ] Better rare-species handling
- [ ] Confidence calibration
- [ ] Advanced model interpretability
- [ ] Cloud deployment
- [ ] Persistent prediction history
- [ ] Enhanced bird information pages
- [ ] Mobile deployment
💡 Project Highlights
🐦 114 Bird Species
🎧 Audio-Based AI Classification
🧠 CNN Deep Learning Model
🎛️ MFCC Feature Engineering
⚡ FastAPI Inference API
⚛️ React Frontend
📊 Interactive 2D Visualization
🌌 Interactive 3D Visualization
🔝 Top-5 Predictions
🧪 V1 → V6 Model Experiments
📈 Held-Out Model Evaluation
⚠️ Limitations
BirdSense-AI is currently an experimental AI project.
The final V6 model achieved:
52.90% held-out test accuracy

Performance varies substantially across species because the dataset contains class imbalance and some species have relatively few examples.
The system should therefore be treated as an AI-assisted classification tool rather than a definitive bird identification system.
🔮 Future Vision
BirdSense-AI can eventually evolve into a more complete acoustic intelligence platform:
                    🎧
                Bird Audio
                     │
                     ▼
             ┌───────────────┐
             │ Audio Engine  │
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     🐦 Species AI        📊 Acoustic AI
          │                     │
          ▼                     ▼
     Identification        Sound Analysis
          │                     │
          └──────────┬──────────┘
                     ▼
              🌿 Bird Intelligence
Future versions could combine stronger audio models, real-time microphone inference, richer species information, and cloud deployment.
🤝 Contributing
Contributions, ideas, and improvements are welcome.
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request
👩‍💻 Developer
<div align="center">

Anjali Saini
B.Tech CSE — AI & Data Science
<a href="https://github.com/anjalisaini89">
  <img src="https://img.shields.io/badge/GitHub-anjalisaini89-181717?style=for-the-badge&logo=github"/>
</a>

<a href="https://www.linkedin.com/in/anjali-saini-598ba0325/">
  <img src="https://img.shields.io/badge/LinkedIn-Anjali%20Saini-0A66C2?style=for-the-badge&logo=linkedin"/>
</a>

</div>

<div align="center">

⭐ If you found BirdSense-AI interesting, consider giving it a star!
From birdsong → to data → to intelligence. 🐦🌌
</div>
```