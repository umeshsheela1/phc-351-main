# Model Testing Script for Speech Emotion Recognition
# This script loads the trained model and tests it on new audio files

import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import argparse
from pathlib import Path

class EmotionPredictor:
    def __init__(self, model_path='model/emotion_recognition_model.h5', 
                 scaler_path='model/scaler.pkl', 
                 encoder_path='model/label_encoder.pkl'):
        """
        Initialize the emotion predictor with trained model and preprocessors
        """
        self.model = tf.keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Model and preprocessors loaded successfully!")
        print(f"Available emotions: {list(self.label_encoder.classes_)}")
    
    def extract_features(self, y, sr):
        """Extract the same features used during training"""
        features = []
        
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
    
    def predict_emotion(self, audio_file_path):
        """
        Predict emotion from audio file
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            tuple: (predicted_emotion, confidence_score, all_probabilities)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=22050, duration=3.0)
            
            # Extract features
            features = self.extract_features(y, sr)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Check if model expects LSTM input shape
            if len(self.model.input_shape) == 3:  # LSTM model
                features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
            
            # Make prediction
            predictions = self.model.predict(features_scaled, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get all probabilities
            all_probabilities = {}
            for i, emotion in enumerate(self.label_encoder.classes_):
                all_probabilities[emotion] = predictions[0][i]
            
            return predicted_emotion, confidence, all_probabilities
            
        except Exception as e:
            print(f"Error predicting emotion for {audio_file_path}: {e}")
            return None, None, None
    
    def test_single_file(self, audio_file_path):
        """Test prediction on a single audio file"""
        print(f"\nTesting file: {audio_file_path}")
        print("-" * 50)
        
        if not os.path.exists(audio_file_path):
            print(f"Error: File {audio_file_path} does not exist!")
            return
        
        emotion, confidence, probabilities = self.predict_emotion(audio_file_path)
        
        if emotion is not None:
            print(f"Predicted Emotion: {emotion.upper()}")
            print(f"Confidence: {confidence:.4f}")
            print("\nAll Probabilities:")
            for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emo}: {prob:.4f}")
        else:
            print("Failed to predict emotion!")
    
    def test_directory(self, directory_path):
        """Test prediction on all audio files in a directory"""
        print(f"\nTesting all files in directory: {directory_path}")
        print("=" * 60)
        
        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} does not exist!")
            return
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(directory_path).glob(f"**/*{ext}"))
        
        if not audio_files:
            print("No audio files found in the directory!")
            return
        
        results = []
        for audio_file in audio_files:
            emotion, confidence, probabilities = self.predict_emotion(str(audio_file))
            if emotion is not None:
                results.append({
                    'file': audio_file.name,
                    'emotion': emotion,
                    'confidence': confidence
                })
                print(f"{audio_file.name}: {emotion} ({confidence:.4f})")
        
        # Summary statistics
        if results:
            print(f"\nSummary:")
            print(f"Total files processed: {len(results)}")
            emotions_count = {}
            for result in results:
                emotion = result['emotion']
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
            
            print("Emotion distribution:")
            for emotion, count in sorted(emotions_count.items()):
                print(f"  {emotion}: {count} files")
            
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"Average confidence: {avg_confidence:.4f}")
    
    def batch_test_with_ground_truth(self, test_csv_path):
        """
        Test on a CSV file with ground truth labels
        
        CSV format: file_path, true_emotion
        """
        print(f"\nTesting with ground truth from: {test_csv_path}")
        print("=" * 60)
        
        try:
            import pandas as pd
            df = pd.read_csv(test_csv_path)
            
            if 'file_path' not in df.columns or 'true_emotion' not in df.columns:
                print("Error: CSV must have 'file_path' and 'true_emotion' columns!")
                return
            
            correct_predictions = 0
            total_predictions = 0
            detailed_results = []
            
            for idx, row in df.iterrows():
                file_path = row['file_path']
                true_emotion = row['true_emotion']
                
                predicted_emotion, confidence, _ = self.predict_emotion(file_path)
                
                if predicted_emotion is not None:
                    is_correct = predicted_emotion == true_emotion
                    correct_predictions += is_correct
                    total_predictions += 1
                    
                    detailed_results.append({
                        'file': os.path.basename(file_path),
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {os.path.basename(file_path)}: {true_emotion} -> {predicted_emotion} ({confidence:.4f})")
            
            # Calculate accuracy
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"\nOverall Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
                
                # Per-emotion accuracy
                from collections import defaultdict
                emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                
                for result in detailed_results:
                    true_emo = result['true_emotion']
                    emotion_stats[true_emo]['total'] += 1
                    if result['correct']:
                        emotion_stats[true_emo]['correct'] += 1
                
                print("\nPer-emotion accuracy:")
                for emotion in sorted(emotion_stats.keys()):
                    stats = emotion_stats[emotion]
                    emo_accuracy = stats['correct'] / stats['total']
                    print(f"  {emotion}: {emo_accuracy:.4f} ({stats['correct']}/{stats['total']})")
            
        except ImportError:
            print("Error: pandas is required for CSV testing. Install with: pip install pandas")
        except Exception as e:
            print(f"Error processing CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Speech Emotion Recognition Model')
    parser.add_argument('--model', default='model/emotion_recognition_model.h5', 
                       help='Path to the trained model')
    parser.add_argument('--scaler', default='model/scaler.pkl', 
                       help='Path to the scaler file')
    parser.add_argument('--encoder', default='model/label_encoder.pkl', 
                       help='Path to the label encoder file')
    parser.add_argument('--file', help='Path to a single audio file to test')
    parser.add_argument('--directory', help='Path to directory containing audio files')
    parser.add_argument('--csv', help='Path to CSV file with ground truth labels')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = EmotionPredictor(args.model, args.scaler, args.encoder)
        
        if args.file:
            # Test single file
            predictor.test_single_file(args.file)
        elif args.directory:
            # Test directory
            predictor.test_directory(args.directory)
        elif args.csv:
            # Test with ground truth
            predictor.batch_test_with_ground_truth(args.csv)
        else:
            # Interactive mode
            print("\nInteractive Testing Mode")
            print("Enter the path to an audio file (or 'quit' to exit):")
            
            while True:
                file_path = input("\nAudio file path: ").strip()
                if file_path.lower() in ['quit', 'exit', 'q']:
                    break
                
                if file_path:
                    predictor.test_single_file(file_path)
    
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure you have trained the model first and all required files exist.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Speech Emotion Recognition - Model Testing")
        print("=" * 50)
        print("Usage examples:")
        print("1. Test single file:")
        print("   python test_model.py --file path/to/audio.wav")
        print("\n2. Test directory:")
        print("   python test_model.py --directory path/to/audio/files/")
        print("\n3. Test with ground truth CSV:")
        print("   python test_model.py --csv path/to/test_data.csv")
        print("\n4. Interactive mode:")
        print("   python test_model.py")
        print("\nNote: Make sure the model files exist:")
        print("- best_emotion_model.h5")
        print("- scaler.pkl") 
        print("- label_encoder.pkl")
    
    main()