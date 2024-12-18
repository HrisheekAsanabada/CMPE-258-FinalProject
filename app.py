import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import pickle

def extract_features(file_path):
    try:
        # Load audio with same parameters as original training
        data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
        
        # Extract feature set matching the training data
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        mfcc = np.ravel(mfcc.T)
        
        zcr = librosa.feature.zero_crossing_rate(y=data)
        zcr = np.ravel(zcr)
        
        rms = librosa.feature.rms(y=data)
        rms = np.ravel(rms)
        
        # Combine features
        features = np.concatenate([zcr, rms, mfcc])
        
        # Pad to ensure consistent length
        features = np.pad(features, (0, 1620 - len(features)), 'constant')
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def get_predict_feat(path):
    # Extract features
    features = extract_features(path)
    
    if features is None:
        return None, None
    
    # Visualize waveform
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Loaded Audio Segment")
    plt.show()
    
    # Load preprocessors
    with open('/Users/likhithaparuchuri/projects/speechApp/scaler2.pickle', 'rb') as f:
        scaler2 = pickle.load(f)
    
    with open('/Users/likhithaparuchuri/projects/speechApp/encoder2.pickle', 'rb') as f:
        encoder2 = pickle.load(f)
    
    # Reshape and scale
    features_reshaped = features.reshape(1, -1)
    i_result = scaler2.transform(features_reshaped)
    final_result = np.expand_dims(i_result, axis=2)

    return final_result, encoder2

@st.cache_resource
def load_emotion_model():
    try:
        # Load the model directly using Keras
        model = tf.keras.models.load_model('/Users/likhithaparuchuri/projects/speechApp/CNN_model.keras')
        print("Loaded model from disk")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit UI
st.title("Speech Emotion Recognition")
st.write("Upload an audio file to detect the emotion!")

model = load_emotion_model()

if model:
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(current_dir, "temp_audio.wav")
        
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display audio player
        st.audio(temp_path)
        
        # Extract features and predict
        features, encoder2 = get_predict_feat(temp_path)
        
        if features is not None:
            emotions1 = {1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust', 8:'Surprise'}
            
            predictions = model.predict(features)
            
            # Get the categories from the encoder
            categories = encoder2.categories_[0]
            
            # Get the predicted emotion
            y_pred = [categories[pred.argmax()] for pred in predictions]
            
            st.success(f"Detected Emotion: {y_pred[0].capitalize()}")
            
        # Cleanup
        os.remove(temp_path)
else:
    st.error("Failed to load the model. Please check the model file.")