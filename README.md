# 🎤 Speech Emotion Recognition System

A comprehensive deep learning-based system for real-time emotion recognition from speech audio using Convolutional Neural Networks (CNNs) and advanced audio feature extraction techniques.

## 🎯 Project Overview

This project implements an end-to-end Speech Emotion Recognition (SER) system capable of classifying human emotions from audio recordings. The system leverages deep learning techniques to analyze acoustic features and predict emotional states with high accuracy across 8 different emotion categories.
[Deployed Link](https://speech-emotion-recognize.streamlit.app/)

### Key Features

- **Real-time emotion detection** from audio files
- **Interactive web interface** built with Streamlit
- **Advanced audio preprocessing** with feature extraction pipeline
- **CNN-based deep learning model** trained on diverse speech datasets
- **Comprehensive visualization** including waveforms, spectrograms, and probability distributions
- **Multi-format audio support** (WAV, MP3, FLAC, AAC, OGG)

## 📊 Supported Emotions

The system can classify the following 8 emotional states:

- 😠 **Angry** - Intense, hostile speech
- 😌 **Calm** - Peaceful, relaxed speech
- 🤢 **Disgust** - Expression of distaste
- 😨 **Fearful** - Anxious, scared speech
- 😊 **Happy** - Joyful, positive speech
- 😐 **Neutral** - Emotionally neutral speech
- 😢 **Sad** - Sorrowful, melancholic speech
- 😲 **Surprised** - Astonished, amazed speech

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/umeshsheela1/phc-351.git
cd Speech-Emotion-Recognition
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
streamlit run app.py
```

### Required Dependencies

```
streamlit
numpy
librosa
tensorflow
plotly
pandas
soundfile
scikit-learn
matplotlib
seaborn
```

## 📈 Dataset Analysis & Preprocessing

### Dataset Exploration

The model was trained on a comprehensive speech emotion dataset with the following characteristics:

#### Data Distribution Analysis

- **Emotion Distribution**: Balanced dataset across 8 emotion categories
- **Gender Distribution**: Equal representation of male and female speakers
- **Source Distribution**: Mix of speech and song-based emotional expressions
- **Intensity Levels**: Varying emotional intensity from subtle to strong
- **Cross-feature Analysis**: Emotion patterns analyzed across gender and source types

### Preprocessing Pipeline

#### 1. Audio Preprocessing

```python
# Audio loading and normalization
audio_data, sample_rate = librosa.load(file_path, sr=22050, duration=3.0)
audio_data = librosa.util.normalize(audio_data)
```

#### 2. Data Augmentation

Multiple augmentation techniques were applied to increase dataset diversity:

- **Noise Addition**: Random noise injection to improve robustness
- **Time Stretching**: Speed variation while preserving pitch
- **Pitch Shifting**: Fundamental frequency modification
- **Time Shifting**: Temporal offset variations
- **Volume Scaling**: Amplitude adjustments

#### 3. Feature Extraction

Comprehensive feature extraction pipeline yielding 200+ acoustic features:

**Spectral Features:**

- Zero Crossing Rate (ZCR) - 4 statistics
- Root Mean Square (RMS) Energy - 4 statistics
- Spectral Centroid - 4 statistics
- Spectral Bandwidth - 4 statistics
- Spectral Rolloff - 4 statistics
- Spectral Contrast - 14 features (7 bands × 2 statistics)
- Spectral Flatness - 2 statistics

**Cepstral Features:**

- Mel-Frequency Cepstral Coefficients (MFCCs) - 80 features (20 × 4 statistics)
- Delta MFCCs (1st derivative) - 40 features
- Delta-Delta MFCCs (2nd derivative) - 40 features

**Harmonic Features:**

- Chroma Features - 24 features (12 × 2 statistics)
- Tonnetz (Tonal Centroid) - 12 features (6 × 2 statistics)

**Temporal Features:**

- Tempo and Beat Tracking - 3 features
- Fundamental Frequency (F0) - 4 features

**Mel-Spectrogram Features:**

- Statistical measures (mean, std, max, min, median, percentiles) - 7 features
- Band energy analysis - 8 features

#### 4. Spectrogram Analysis

- **Mel-spectrograms** generated for visual pattern recognition
- **STFT spectrograms** for frequency domain analysis
- **Chromagram** for harmonic content analysis

## 🧠 Model Architecture

### CNN Model Design

The emotion recognition system employs a Convolutional Neural Network optimized for audio feature classification:

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')  # 8 emotion classes
])
```

### Training Configuration

- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 100 with early stopping
- **Validation Split**: 20%
- **Regularization**: L2 regularization + Dropout layers

### Model Pipeline

1. **Feature Extraction**: Raw audio → 200+ dimensional feature vector
2. **Normalization**: StandardScaler for feature scaling
3. **Classification**: CNN model for emotion prediction
4. **Post-processing**: Probability distribution and confidence scoring

## 📊 Model Performance

### CNN Model Results

- **Overall Accuracy**: 85.52%
- **F1 Score**: 85.50%
- **Total Test Samples**: 1,962

### Classification Report

```
              precision    recall  f1-score   support
angry            0.92      0.91      0.91       301
calm             0.89      0.90      0.90       301
disgust          0.80      0.83      0.81       153
fearful          0.84      0.80      0.82       301
happy            0.88      0.85      0.87       301
neutral          0.81      0.91      0.86       150
sad              0.80      0.78      0.79       301
surprised        0.85      0.89      0.87       154

accuracy                           0.86      1962
macro avg        0.85      0.86      0.85      1962
weighted avg     0.86      0.86      0.86      1962
```

### Confusion Matrix

The model demonstrates strong performance across all emotion categories with minimal cross-class confusion, particularly excelling in happy and calm emotion recognition.
![Confusion matrix](./assets/confusionmatrix.png)

### Performance Insights

- **Best Performance**: Angry emotion (92% precision, 91% recall)
- **Strong Performers**: Calm (89% precision, 90% recall) and Happy (88% precision, 85% recall)
- **Challenging Classes**: Disgust and Sad emotions showing room for improvement
- **Balanced Dataset**: Consistent support across most emotion categories (301 samples)
- **Robust Classification**: 85.52% overall accuracy demonstrates reliable emotion detection

## 🌐 Web Application Features

### Interactive Interface

- **File Upload**: Support for multiple audio formats
- **Real-time Processing**: Instant emotion prediction
- **Audio Playback**: Built-in audio player
- **Visual Analytics**: Multiple visualization options

### Visualization Components

1. **Waveform Display**: Time-domain audio representation
2. **Spectrogram**: Frequency-time analysis
3. **Probability Distribution**: Confidence scores for all emotions
4. **Feature Analysis**: Technical audio characteristics

### User Experience

- **Responsive Design**: Modern gradient-based UI
- **Progress Indicators**: Real-time processing feedback
- **Confidence Scoring**: Reliability assessment
- **Detailed Analytics**: Comprehensive audio analysis

## 🧪 Testing & Evaluation

### Testing Script Usage

The project includes a comprehensive testing script (`testing_script.py`) for evaluating the trained model on new audio data.

#### Command Line Interface

```bash
python testing_script.py [OPTIONS]
```

#### Available Options

- `--model`: Path to the trained model (default: `model/best_emotion_recognition_model.h5`)
- `--scaler`: Path to the scaler file (default: `model/scaler.pkl`)
- `--encoder`: Path to the label encoder file (default: `model/label_encoder.pkl`)
- `--file`: Path to a single audio file to test
- `--directory`: Path to directory containing multiple audio files
- `--csv`: Path to CSV file with ground truth labels for batch evaluation

#### Example Usage

```bash
# Test a single audio file
python testing_script.py --file path/to/audio.wav

# Test multiple files in a directory
python testing_script.py --directory path/to/audio/folder

# Batch evaluation with ground truth labels
python testing_script.py --directory path/to/audio/folder --csv path/to/labels.csv

# Using custom model paths
python testing_script.py --model custom/model.h5 --scaler custom/scaler.pkl --encoder custom/encoder.pkl --file test.wav
```

## 🚀 Deployment

### Local Deployment

```bash
streamlit run app.py
```

## 📁 Project Structure

```
speech-emotion-recognition/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── testing_script.py              # Testing script for model evaluation
├── .gitignore                     # Git ignore file
├── model/                         # Trained model artifacts
│   ├── best_emotion_recognition_model.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── notebooks/                     # Jupyter notebooks
│   └── Speech Emotion Recognition Notebook.ipynb
├── venv/                         # Virtual environment
└── assets/
```

## 🔬 Technical Implementation

### Audio Processing

- **Sampling Rate**: 22.05 kHz standardization
- **Duration**: 3-second audio segments
- **Preprocessing**: Noise reduction and normalization
- **Feature Engineering**: Multi-domain feature extraction

### Model Training Process

1. **Data Loading**: Batch processing of audio files
2. **Feature Extraction**: Parallel processing for efficiency
3. **Data Augmentation**: Synthetic data generation
4. **Model Training**: CNN with regularization
5. **Model Selection**: Best performing model based on validation accuracy

### Optimization Techniques

- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Batch Normalization**: Stable training process
- **Dropout Regularization**: Improved generalization

## 🎯 Future Enhancements

### Planned Features

- **Real-time Microphone Input**: Live emotion detection
- **Multi-language Support**: Emotion recognition across languages
- **Emotion Intensity Scoring**: Granular emotion strength measurement
- **Speaker Identification**: Individual speaker emotion patterns
- **Emotion Transition Analysis**: Temporal emotion changes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔗 References

- Librosa: Audio analysis library
- TensorFlow: Deep learning framework
- Streamlit: Web application framework
- Plotly: Interactive visualizations

