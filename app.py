import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import tempfile
import os
from io import BytesIO
import soundfile as sf

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        color: #2E8B57;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_emotion_model(model_path='model/emotion_recognition_model.h5', 
                      scaler_path='model/scaler.pkl', 
                      encoder_path='model/label_encoder.pkl'):
    """Load model and preprocessors with caching"""
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

class StreamlitEmotionPredictor:
    def __init__(self):
        # Initialize with session state to persist across reruns
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.label_encoder = None
    
    def load_model_components(self):
        """Load model components and store in session state"""
        model, scaler, label_encoder, success = load_emotion_model()
        if success:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.label_encoder = label_encoder
            st.session_state.model_loaded = True
        return success
    
    @property
    def model_loaded(self):
        return st.session_state.get('model_loaded', False)
    
    @property
    def model(self):
        return st.session_state.get('model', None)
    
    @property
    def scaler(self):
        return st.session_state.get('scaler', None)
    
    @property
    def label_encoder(self):
        return st.session_state.get('label_encoder', None)
    
    def extract_features(self, y, sr):
        """Extract audio features"""
        features = []
        
        try:
            # 1. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
            
            # 2. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
            
            # 3. Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid), 
                            np.max(spectral_centroid), np.min(spectral_centroid)])
            
            # 4. Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                            np.max(spectral_bandwidth), np.min(spectral_bandwidth)])
            
            # 5. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff),
                            np.max(spectral_rolloff), np.min(spectral_rolloff)])
            
            # 6. Enhanced MFCCs (20 coefficients instead of 13)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features.extend([np.mean(mfccs[i]), np.std(mfccs[i]), 
                                np.max(mfccs[i]), np.min(mfccs[i])])
            
            # 7. Delta MFCCs (first derivative)
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(20):
                features.extend([np.mean(delta_mfccs[i]), np.std(delta_mfccs[i])])
            
            # 8. Delta-Delta MFCCs (second derivative)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            for i in range(20):
                features.extend([np.mean(delta2_mfccs[i]), np.std(delta2_mfccs[i])])
            
            # 9. Chroma features (enhanced)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            for i in range(12):
                features.extend([np.mean(chroma[i]), np.std(chroma[i])])
            
            # 10. Tonnetz (enhanced)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            for i in range(6):
                features.extend([np.mean(tonnetz[i]), np.std(tonnetz[i])])
            
            # 11. Spectral Contrast (enhanced)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            for i in range(7):  # n_bands + 1
                features.extend([np.mean(spectral_contrast[i]), np.std(spectral_contrast[i])])
            
            # 12. Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            
            # Statistical features from mel spectrogram
            features.extend([
                np.mean(mel_spec_db), np.std(mel_spec_db), 
                np.max(mel_spec_db), np.min(mel_spec_db),
                np.median(mel_spec_db), np.percentile(mel_spec_db, 25),
                np.percentile(mel_spec_db, 75)
            ])
            
            # Mel spectrogram band energy
            for i in range(0, 128, 16):  # 8 bands
                band_energy = np.mean(mel_spec_db[i:i+16])
                features.append(band_energy)
            
            # 13. Pitch and Harmony Features
            # Fundamental frequency (F0)
            try:
                f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                        fmax=librosa.note_to_hz('C7'))
                if f0 is not None and len(f0) > 0:
                    f0 = np.array(f0).flatten()
                    voiced_flag = np.array(voiced_flag).flatten()
                    f0_clean = f0[voiced_flag & ~np.isnan(f0)]
                    
                    if len(f0_clean) > 0:
                        features.extend([float(np.mean(f0_clean)), float(np.std(f0_clean)), 
                                        float(np.max(f0_clean)), float(np.min(f0_clean))])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            except Exception as e:
                print(f"Warning: Pitch extraction failed: {e}")
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 14. Tempo and Rhythm Features
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))
                
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    features.extend([float(np.mean(beat_intervals)), float(np.std(beat_intervals))])
                else:
                    features.extend([0.0, 0.0])
            except Exception as e:
                print(f"Warning: Tempo extraction failed: {e}")
                features.extend([0.0, 0.0, 0.0])
            
            # 15. Spectral Features
            try:
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                features.extend([float(np.mean(spectral_flatness)), float(np.std(spectral_flatness))])
            except Exception as e:
                print(f"Warning: Spectral flatness extraction failed: {e}")
                features.extend([0.0, 0.0])
            
            try:
                poly_features = librosa.feature.poly_features(y=y, sr=sr, order=1)
                if poly_features.ndim == 2 and poly_features.shape[0] > 0:
                    poly_coeff = poly_features[0]
                    features.extend([float(np.mean(poly_coeff)), float(np.std(poly_coeff))])
                else:
                    features.extend([0.0, 0.0])
            except Exception as e:
                print(f"Warning: Poly features extraction failed: {e}")
                features.extend([0.0, 0.0])
            
            features_array = np.array(features, dtype=np.float32)
            
            if features_array.ndim != 1:
                raise ValueError(f"Features array should be 1D, got shape: {features_array.shape}")
            
            return features_array
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def predict_emotion(self, audio_data, sr):
        """Predict emotion from audio data"""
        try:
            features = self.extract_features(audio_data, sr)
            if features is None:
                return None, None, None
            
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            if len(self.model.input_shape) == 3:
                features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
            
            predictions = self.model.predict(features_scaled, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            all_probabilities = {}
            for i, emotion in enumerate(self.label_encoder.classes_):
                all_probabilities[emotion] = predictions[0][i]
            
            return predicted_emotion, confidence, all_probabilities
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None, None, None

def plot_waveform(audio_data, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, 
        y=audio_data,
        mode='lines',
        name='Amplitude',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_spectrogram(audio_data, sr, title="Spectrogram"):
    """Plot spectrogram"""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig

def plot_emotion_probabilities(probabilities):
    """Plot emotion probabilities as a bar chart"""
    emotions = list(probabilities.keys())
    probs = [probabilities[emotion] for emotion in emotions]
    
    colors = {
        'angry': '#FF6B6B',
        'calm': '#4ECDC4', 
        'disgust': '#95A5A6',
        'fearful': '#9B59B6',
        'happy': '#F39C12',
        'neutral': '#34495E',
        'sad': '#3498DB',
        'surprised': '#E67E22'
    }
    
    bar_colors = [colors.get(emotion, '#BDC3C7') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=bar_colors,
            text=[f'{p:.3f}' for p in probs],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Emotion Prediction Probabilities',
        xaxis_title='Emotions',
        yaxis_title='Probability',
        height=500,
        showlegend=False
    )
    
    return fig

def main():
    # Initialize predictor
    predictor = StreamlitEmotionPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an audio file to detect the emotion in speech")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Model Information")
        
        # Model loading
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                success = predictor.load_model_components()
                if success:
                    st.success("✅ Model loaded successfully!")
                    st.rerun()  # Force refresh to update UI
                else:
                    st.error("❌ Failed to load model")
        
        if predictor.model_loaded:
            st.success("🟢 Model Status: Ready")
            st.info(f"📊 Emotions: {', '.join(predictor.label_encoder.classes_)}")
        else:
            st.warning("🟡 Model Status: Not Loaded")
            st.info("Click 'Load Model' to start")
        
        st.markdown("---")
        st.markdown("## ℹ️ Instructions")
        st.markdown("""
        1. Load the model first
        2. Upload an audio file (.wav, .mp3, .flac)
        3. Wait for processing
        4. View results and visualizations
        
        **Supported emotions:**
        - 😠 Angry
        - 😌 Calm  
        - 🤢 Disgust
        - 😨 Fearful
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        - 😲 Surprised
        """)
    
    # Main content
    if not predictor.model_loaded:
        st.info("👈 Please load the model first using the sidebar")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'aac', 'ogg'],
        help="Upload an audio file to analyze emotions"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎵 Audio Analysis")
            
            # Display file info
            st.info(f"📁 **File:** {uploaded_file.name} | **Size:** {uploaded_file.size} bytes")
            
            # Play audio
            st.audio(uploaded_file, format='audio/wav')
            
            # Process audio
            with st.spinner("🔄 Processing audio..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load audio
                    audio_data, sr = librosa.load(tmp_file_path, sr=22050, duration=3.0)
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    # Make prediction
                    emotion, confidence, probabilities = predictor.predict_emotion(audio_data, sr)
                    
                    if emotion is not None:
                        # Display main result
                        with col2:
                            st.markdown("### 🎯 Prediction Results")
                            
                            # Emotion card
                            st.markdown(f"""
                            <div class="emotion-card">
                                <h2>🎭 {emotion.upper()}</h2>
                                <div class="confidence-score">{confidence:.2%}</div>
                                <p>Confidence Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar for confidence
                            st.progress(float(confidence))
                            
                            # Confidence interpretation
                            if confidence > 0.8:
                                st.success("🎯 High Confidence")
                            elif confidence > 0.6:
                                st.warning("⚠️ Medium Confidence")
                            else:
                                st.error("❓ Low Confidence")
                        
                        # Visualizations
                        st.markdown("---")
                        st.markdown("### 📊 Detailed Analysis")
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3, tab4 = st.tabs(["🎵 Waveform", "🌈 Spectrogram", "📈 Probabilities", "📋 Details"])
                        
                        with tab1:
                            fig_waveform = plot_waveform(audio_data, sr)
                            st.plotly_chart(fig_waveform, use_container_width=True)
                        
                        with tab2:
                            fig_spectrogram = plot_spectrogram(audio_data, sr)
                            st.plotly_chart(fig_spectrogram, use_container_width=True)
                        
                        with tab3:
                            fig_probs = plot_emotion_probabilities(probabilities)
                            st.plotly_chart(fig_probs, use_container_width=True)
                            
                            # Probability table
                            prob_df = pd.DataFrame({
                                'Emotion': list(probabilities.keys()),
                                'Probability': [f"{prob:.4f}" for prob in probabilities.values()],
                                'Percentage': [f"{prob:.2%}" for prob in probabilities.values()]
                            }).sort_values('Probability', ascending=False, key=lambda x: x.astype(float))
                            
                            st.dataframe(prob_df, use_container_width=True)
                        
                        with tab4:
                            st.markdown("#### 🔍 Technical Details")
                            
                            col_detail1, col_detail2 = st.columns(2)
                            
                            with col_detail1:
                                st.metric("📊 Sample Rate", f"{sr} Hz")
                                st.metric("⏱️ Duration", f"{len(audio_data)/sr:.2f} seconds")
                                st.metric("📏 Samples", len(audio_data))
                            
                            with col_detail2:
                                st.metric("🎯 Predicted Class", emotion)
                                st.metric("📈 Max Probability", f"{confidence:.4f}")
                                st.metric("🔢 Total Classes", len(probabilities))
                            
                            # Feature importance (simplified)
                            st.markdown("#### 🧮 Feature Analysis")
                            features = predictor.extract_features(audio_data, sr)
                            if features is not None:
                                st.success(f"✅ Extracted {len(features)} audio features")
                                
                                # Show some key features
                                zcr_mean = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
                                rms_mean = np.mean(librosa.feature.rms(y=audio_data)[0])
                                spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
                                
                                st.write(f"**Zero Crossing Rate:** {zcr_mean:.4f}")
                                st.write(f"**RMS Energy:** {rms_mean:.4f}")
                                st.write(f"**Spectral Centroid:** {spectral_centroid_mean:.2f} Hz")
                    
                    else:
                        st.error("❌ Failed to process audio file")
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
    
    else:
        st.markdown("### 🚀 Try the Demo")
        st.markdown("""
        Upload an audio file to get started! The system can recognize the following emotions:
        
        - **😠 Angry** - Intense, hostile speech
        - **😌 Calm** - Peaceful, relaxed speech  
        - **🤢 Disgust** - Expression of distaste
        - **😨 Fearful** - Anxious, scared speech
        - **😊 Happy** - Joyful, positive speech
        - **😐 Neutral** - Emotionally neutral speech
        - **😢 Sad** - Sorrowful, melancholic speech
        - **😲 Surprised** - Astonished, amazed speech
        """)
        
        # Add some tips
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - Use clear, high-quality audio recordings
            - Ensure the audio contains speech (not just music)
            - Audio length should be at least 1-2 seconds
            - Minimize background noise
            - Supported formats: WAV, MP3, FLAC, AAC, OGG
            """)

if __name__ == "__main__":
    main()