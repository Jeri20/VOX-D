import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained LSTM model
model = load_model('disease_detection_model.h5')

# Define feature extraction functions
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    jitter = compute_jitter(y, sr)
    shimmer = compute_shimmer(y, sr)
    # Additional features (replace with actual feature extraction)
    features = [jitter, shimmer, np.random.random(), np.random.random(), np.random.random()]
    return np.array(features)

def compute_jitter(y, sr):
    # Placeholder jitter calculation (replace with actual method)
    return np.random.random()

def compute_shimmer(y, sr):
    # Placeholder shimmer calculation (replace with actual method)
    return np.random.random()

# Streamlit UI
st.title('Parkinson\'s Disease Detection from Voice')
st.write('Upload a voice recording to analyze for potential Parkinson\'s disease indicators.')

# Upload file
uploaded_file = st.file_uploader("Upload your voice recording", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Save the uploaded file temporarily
    with open('temp_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the audio file
    features = extract_features('temp_audio.wav')

    # Standardize the features using a pre-fit scaler (fit during training)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))

    # Reshape for LSTM model
    features_scaled_reshaped = features_scaled.reshape(1, 1, -1)  # (num_samples, num_timesteps, num_features)

    # Make prediction
    prediction = model.predict(features_scaled_reshaped)
    disease_status = 'Parkinson\'s Disease Detected' if prediction > 0.5 else 'No Parkinson\'s Disease Detected'

    # Display the result
    st.write(f'Prediction: **{disease_status}**')
