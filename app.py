import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import glob

# Load audio files
def load_audio_files(path):
    audio_files = glob.glob(os.path.join(path, '*.wav'))
    return audio_files

# Extract features from audio files
def extract_features(audio_files):
    X = []
    for file in audio_files:
        y, sr = librosa.load(file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        X.append(mfccs)
    return np.array(X)

# Create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train SVM model
def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train_scaled, y_train)
    return svm_model, scaler

# Load and preprocess data
audio_path = 'Laryngozele'  # Replace with the correct path to your audio files
audio_files = load_audio_files(audio_path)
X = extract_features(audio_files)

# Labeling logic: assuming 'a' in filename means laryngozele and 'n' means normal
y = np.array([0 if 'a' in f else 1 for f in audio_files])  # Adjust as needed

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM and CNN
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train models
lstm_model = create_lstm_model((1, X_train.shape[1]))
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_split=0.1)

cnn_model = create_cnn_model((X_train.shape[1], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_split=0.1)

svm_model, scaler = train_svm(X_train, y_train)

# Model selection
model_choice = st.selectbox("Choose Model", ["LSTM", "CNN", "SVM"])

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    if model_choice == "LSTM":
        mfccs_lstm = mfccs.reshape((1, 1, mfccs.shape[1]))
        prediction = lstm_model.predict(mfccs_lstm)
        st.write("Prediction (LSTM):", "Laryngozele" if prediction[0][0] < 0.5 else "Normal")
    elif model_choice == "CNN":
        mfccs_cnn = mfccs.reshape((1, mfccs.shape[1], 1))
        prediction = cnn_model.predict(mfccs_cnn)
        st.write("Prediction (CNN):", "Laryngozele" if prediction[0][0] < 0.5 else "Normal")
    elif model_choice == "SVM":
        mfccs_scaled = scaler.transform(mfccs)
        prediction = svm_model.predict(mfccs_scaled)
        st.write("Prediction (SVM):", "Laryngozele" if prediction[0] == 0 else "Normal")
