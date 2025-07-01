# DeepEmotion
# 😃 DeepEmotion — Facial Emotion Detection with CNN and Dlib

DeepEmotion is a facial emotion recognition system built using deep learning (CNNs) and `dlib` face detection. It supports real-time webcam emotion recognition and is trained on the FER2013 dataset.

---

## 📁 Project Structure

DeepEmotion/
├── data/ # Folder to place the FER2013 dataset
│ └── fer2013(dataset)
├── models/ # Trained model files (.h5)
│ └── ResNet50_model.h5
├── train_model.py # Model training script
├── real_time_detection.py # Real-time webcam detection using dlib + OpenCV
├── data_loader.py # Data loading and preprocessing
├── model.py # CNN model architecture
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


---

## 🔗 Dataset

We use the [FER2013 (Facial Expression Recognition)](https://www.kaggle.com/datasets/msambare/fer2013) dataset from Kaggle.

➡️ Download here:  
https://www.kaggle.com/datasets/msambare/fer2013

After downloading, place the `test and train` files in the `data/` directory:


---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Bhavanapoli08/DeepEmotion.git
cd DeepEmotion

2. Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate

3. Install dependencies
For Apple Silicon (M1/M2):
pip install tensorflow-macos==2.15.0 keras==2.15.0
pip install opencv-python numpy matplotlib dlib

🧠 Train the Model
python train_model.py

This trains a ResNet-based CNN on FER2013 and saves ResNet50_model.h5 inside models/.

🎥 Real-Time Emotion Detection
python real_time_detection.py
It will:

Use dlib for face detection

Use your trained model to classify emotions in real-time





