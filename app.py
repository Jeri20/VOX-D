import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained LSTM model
model = load_model('disease_detection_model.h5')

# Print model summary to verify input shape
#st.write("Model Summary:")
#model.summary(print_fn=lambda x: st.write(x))  # Print model summary in Streamlit

# Define feature extraction functions
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Ensure the extraction logic produces 16 features, not just 5
    features = np.random.random(16)  # Placeholder for actual feature extraction
    return np.array(features)

def compute_jitter(y, sr):
    return np.random.random()

def compute_shimmer(y, sr):
    return np.random.random()

# Streamlit UI
st.title("Disease Detection from Voice")
st.write("Upload a voice recording to analyze for potential Parkinson's disease indicators.")

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
    st.write(f'Extracted Features: {features}')

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))  # Use the saved scaler for consistency
    #st.write(f'Scaled Features: {features_scaled}')

    # Ensure that the reshaped data matches the LSTM input shape
    num_timesteps = 1  # Replace with the actual number of timesteps used during training
    num_features = features_scaled.shape[1]  # Number of features
    features_scaled_reshaped = features_scaled.reshape(1, num_timesteps, num_features).astype('float32')
    st.write(f'Reshaped Features: {features_scaled_reshaped.shape}')

    # Make prediction
    try:
        prediction = model.predict(features_scaled_reshaped)
        st.write(f'Raw Prediction Output: {prediction}')
        disease_status = 'Parkinson\'s Disease Detected' if prediction > 0.5 else 'No Parkinson\'s Disease Detected'
        st.write(f'Prediction: **{disease_status}**')
    except Exception as e:
        st.error(f'Error making prediction: {e}')
