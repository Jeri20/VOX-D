import streamlit as st
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
import os
import glob

# Function to extract features from local audio files
def extract_features(file_path, label):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features, label

# Path to the directories
laryngoze_dir = './Laryngozele/'
normal_dir = './Normal/'

# Collecting file paths
laryngoze_files = glob.glob(os.path.join(laryngoze_dir, '*.wav'))
normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))

# Extract features for all audio files
X = []
y = []

for file_path in laryngoze_files:
    features, label = extract_features(file_path, 0)
    X.append(features)
    y.append(label)

for file_path in normal_files:
    features, label = extract_features(file_path, 1)
    X.append(features)
    y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for LSTM and CNN
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Train LSTM model
def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    model.save('lstm_model.h5')
    return model

# Train CNN model
def train_cnn(X_train, y_train):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    model.save('cnn_model.h5')
    return model

# Train SVM model
def train_svm(X_train, y_train):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return model

# Train the models
lstm_model = train_lstm(X_train_lstm, y_train)
cnn_model = train_cnn(X_train_lstm, y_train)
svm_model = train_svm(X_train, y_train)

# Streamlit UI
st.title("Voice Disorder Detection")
st.write("Upload a voice recording to analyze for potential voice disorders.")

# Upload file
uploaded_file = st.file_uploader("Upload your voice recording", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Save the uploaded file temporarily
    with open('temp_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the audio file
    y, sr = librosa.load('temp_audio.wav', sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_scaled_reshaped = features_scaled.reshape(1, 1, features_scaled.shape[1])

    # Predict with LSTM model
    lstm_prediction = lstm_model.predict(features_scaled_reshaped)
    lstm_disease_status = 'Voice Disorder Detected' if lstm_prediction[0][0] > 0.5 else 'No Voice Disorder Detected'
    st.write(f'LSTM Prediction: **{lstm_disease_status}**')

    # Predict with CNN model
    cnn_prediction = cnn_model.predict(features_scaled_reshaped)
    cnn_disease_status = 'Voice Disorder Detected' if cnn_prediction[0][0] > 0.5 else 'No Voice Disorder Detected'
    st.write(f'CNN Prediction: **{cnn_disease_status}**')

    # Predict with SVM model
    svm_prediction = svm_model.predict(features_scaled)
    svm_disease_status = 'Voice Disorder Detected' if svm_prediction[0] == 1 else 'No Voice Disorder Detected'
    st.write(f'SVM Prediction: **{svm_disease_status}**')

    os.remove('temp_audio.wav')
